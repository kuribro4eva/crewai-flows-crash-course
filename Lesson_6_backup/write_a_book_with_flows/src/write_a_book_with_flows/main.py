#!/usr/bin/env python
import asyncio
import logging
from typing import List
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

from write_a_book_with_flows.types import BookOutline, Chapter, ChapterOutline
from write_a_book_with_flows.crews.write_book_chapter_crew.write_book_chapter_crew import WriteBookChapterCrew
from write_a_book_with_flows.crews.outline_book_crew.outline_crew import OutlineCrew

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BookState(BaseModel):
    title: str = "Synthesized Insights: A Comprehensive Analysis of Source Materials"
    book: List[Chapter] = []
    book_outline: List[ChapterOutline] = []
    topic: str = "Creating a comprehensive synthesis and analysis of the key concepts from the provided source materials"
    goal: str = """
        The goal of this book is to create a well-structured synthesis of the provided PDF source materials.
        The book will extract, organize, and present the most important concepts and insights from these materials
        in a clear and accessible format. Each chapter should maintain fidelity to the source content while
        presenting information in a logical sequence that builds understanding for the reader.
        The final product should serve as both a comprehensive overview and a detailed analysis of the source materials.
    """

class BookFlow(Flow[BookState]):
    @start()
    def generate_book_outline(self):
        logger.info("Kickoff the Book Outline Crew")
        result = OutlineCrew().crew().kickoff(
            inputs={"topic": self.state.topic, "goal": self.state.goal}
        )

        logger.info("Processing outline result")
        try:
            if result.json_dict:
                logger.debug(f"JSON result: {result.json_dict}")
                book_outline = BookOutline(**result.json_dict)
                self.state.book_outline = book_outline.chapters
                logger.info(f"Generated {len(self.state.book_outline)} chapters")
            else:
                logger.error("No JSON output from outline crew")
                raise ValueError("Invalid outline crew output")
        except Exception as e:
            logger.error(f"Failed to process outline: {e}")
            raise

    @listen(generate_book_outline)
    async def write_chapters(self):
        logger.info("Writing Book Chapters")
        tasks = []

        async def write_single_chapter(chapter_outline):
            """Helper function to write a single chapter"""
            try:
                logger.info(f"Writing Chapter: {chapter_outline.title}")
                result = WriteBookChapterCrew().crew().kickoff(
                    inputs={
                        "goal": self.state.goal,
                        "topic": self.state.topic,
                        "chapter_title": chapter_outline.title,
                        "chapter_description": chapter_outline.description,
                        "book_outline": [
                            co.model_dump() for co in self.state.book_outline
                        ],
                    }
                )

                if result.json_dict:
                    logger.debug(f"JSON result for {chapter_outline.title}: {result.json_dict}")
                    chapter = Chapter(**result.json_dict)
                    return chapter
                else:
                    logger.error(f"No JSON output for chapter {chapter_outline.title}")
                    return None

            except Exception as e:
                logger.error(f"Error writing chapter {chapter_outline.title}: {e}")
                return None

        # Create tasks for parallel chapter writing
        for chapter_outline in self.state.book_outline:
            logger.info(f"Scheduling Chapter: {chapter_outline.title}")
            task = asyncio.create_task(write_single_chapter(chapter_outline))
            tasks.append(task)

        # Wait for all chapters to be written
        chapters = await asyncio.gather(*tasks)
        successful_chapters = [c for c in chapters if c is not None]

        if not successful_chapters:
            raise ValueError("No chapters were successfully generated")

        logger.info(f"Successfully generated {len(successful_chapters)} chapters")
        self.state.book.extend(successful_chapters)

    @listen(write_chapters)
    async def join_and_save_chapter(self):
        logger.info("Joining and Saving Book Chapters")
        book_content = ""

        for chapter in self.state.book:
            book_content += f"# {chapter.title}\n\n"
            book_content += f"{chapter.content}\n\n"

        filename = f"./{self.state.title.replace(' ', '_')}.md"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(book_content)

        logger.info(f"Book saved as {filename}")

def kickoff():
    flow = BookFlow()
    flow.kickoff()

def plot():
    flow = BookFlow()
    flow.plot()

if __name__ == "__main__":
    kickoff()
