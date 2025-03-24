import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from pydantic import Field

from app.agent.browser import BrowserAgent
from app.config import config
from app.prompt.browser import NEXT_STEP_PROMPT as BROWSER_NEXT_STEP_PROMPT
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message, ToolCall
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.focused_documentation_tool import FocusedDocumentationTool
from app.tool.file_saver import FileSaver
from app.tool.python_execute import PythonExecute
from app.tool.web_search import WebSearch
from app.tool.mcp_agent_tool import MCPAgentTool
from app.logger import logger

class Manus(BrowserAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends BrowserAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # Project ID for documentation
    active_project_id: Optional[str] = None
    current_step_index: Optional[int] = None
    past_step_indices: List[int] = Field(default_factory=list)
    collected_urls: List[str] = Field(default_factory=list)

    # Track the current thought index for linking actions to thoughts
    current_thought_index: Optional[int] = None

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            WebSearch(),
            FileSaver(),
            FocusedDocumentationTool(),
            MCPAgentTool(),
            Terminate()
        )
    )

    async def run(self, request: Optional[str] = None) -> str:
        """Override run to automatically document the process"""
        # Initialize documentation project when starting a new request
        if request:
            self.active_project_id = f"project_{int(time.time())}"

            # Initialize documentation project
            await self._init_documentation(request)

        result = await super().run(request)

        # Document completion
        if self.active_project_id:
            await self._document_completion(result)

        return result

    async def think(self) -> bool:
        """Override think to document agent thoughts"""
        # Get current message to use as thought content
        thought_content = None
        if self.messages and len(self.messages) > 0:
            last_message = self.messages[-1]
            thought_content = last_message.content

        logger.info(f"Manus agent think() called, current_step_index={self.current_step_index}, active_project_id={self.active_project_id}")

        # Call parent implementation
        result = await super().think()

        # Document thought if we have an active project and content
        if result and self.active_project_id and thought_content:
            # The key fix: If current_step_index is None, use 0 as a fallback
            # This happens when PlanningFlow hasn't set the step index yet
            step_index = self.current_step_index
            if step_index is None:
                logger.warning("current_step_index is None, using 0 as fallback for documentation")
                step_index = 0

            logger.info(f"Documenting thought for step {step_index}")
            await self._document_thought_with_index(thought_content, step_index)

        return result

    async def _document_thought_with_index(self, thought_content: str, step_index: int) -> None:
        """Document the agent's thought process with explicit step index"""
        if not self.active_project_id or not thought_content:
            logger.warning("Skipping _document_thought: missing project ID or thought content")
            return

        logger.info(f"_document_thought_with_index called with step_index={step_index}, content length={len(thought_content)}")

        try:
            result = await self.available_tools.execute(
                name="documentation",
                tool_input={
                    "command": "document_thought",
                    "project_id": self.active_project_id,
                    "step_index": step_index,
                    "content": thought_content
                }
            )

            logger.info(f"Thought documented for step {step_index}, result: {result}")

            # Store the thought index for use when documenting actions
            # Check if result is a dictionary and has the expected structure
            if result and hasattr(result, "get") and result.get("success") and "thought_index" in result:
                self.current_thought_index = result["thought_index"]
                logger.info(f"Updated current_thought_index to {self.current_thought_index}")
            else:
                # Handle case where result is not a dictionary or missing expected keys
                logger.warning(f"Couldn't update thought index: invalid result structure: {type(result)}")
        except Exception as e:
            logger.error(f"Error documenting thought: {str(e)}")
            logger.exception("Detailed exception:")

    async def act(self) -> str:
        """Override act to document actions and results"""
        # Store the tool calls before execution for documentation
        current_tool_calls = self.tool_calls.copy() if self.tool_calls else []

        logger.info(f"Manus agent act() called, current_step_index={self.current_step_index}, active_project_id={self.active_project_id}")
        if current_tool_calls:
            logger.info(f"Tool calls: {[t.function.name if t and t.function else 'unknown' for t in current_tool_calls]}")

        # Call parent implementation
        result = await super().act()

        # Document action if we have tool calls
        if self.active_project_id and current_tool_calls:
            # Similar fix for act(): If current_step_index is None, use 0 as fallback
            step_index = self.current_step_index
            if step_index is None:
                logger.warning("current_step_index is None, using 0 as fallback for documentation")
                step_index = 0

            logger.info(f"Documenting action for step {step_index}")
            await self._document_action_with_index(current_tool_calls, result, step_index)

            # Collect URLs when using browser
            await self._collect_browser_urls(current_tool_calls)

            # Save collected URLs if any
            if self.collected_urls:
                await self._save_collected_urls()

        return result

    async def _document_action_with_index(self, tool_calls: List[ToolCall], result: str, step_index: int) -> None:
        """Document an action and its result with explicit step index"""
        if not self.active_project_id or not tool_calls:
            logger.warning("Skipping _document_action: missing project ID or tool calls")
            return

        # Get the primary tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name if tool_call and tool_call.function else "unknown"

        logger.info(f"_document_action_with_index called with step_index={step_index}, tool={tool_name}")

        # Don't document documentation tool calls to avoid recursion
        if tool_name == "documentation":
            logger.info("Skipping documentation of a documentation tool call (avoiding recursion)")
            return

        # Format the action content
        try:
            import json
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            action_content = f"Tool: {tool_name}\nArguments: ```json\n{json.dumps(args, indent=2)}\n```\n\nResult:\n```\n{result}\n```"

            action_result = await self.available_tools.execute(
                name="documentation",
                tool_input={
                    "command": "document_action",
                    "project_id": self.active_project_id,
                    "step_index": step_index,
                    "content": action_content,
                    "metadata": {
                        "tool_name": tool_name,
                        "tool_id": tool_call.id,
                        "arguments": args
                    }
                }
            )

            logger.info(f"Action with tool '{tool_name}' documented for step {step_index}, result: {action_result}")

            # If the tool is a browser tool and returned HTML content, save it
            if tool_name == "browser_use" and result and isinstance(result, str) and "<html" in result.lower():
                await self._save_html_content_with_index(result, f"Browser content from {tool_name}", step_index)

        except Exception as e:
            logger.error(f"Error documenting action: {str(e)}")
            logger.exception("Detailed exception:")

    async def _save_html_content_with_index(self, content: str, title: str, step_index: int) -> None:
        """Save HTML content from browsing with explicit step index"""
        if not self.active_project_id:
            return

        logger.info(f"_save_html_content_with_index called with step_index={step_index}")

        # Safety check for content type
        if not isinstance(content, str):
            logger.warning(f"Skipping _save_html_content_with_index: content is not a string but {type(content)}")
            return

        try:
            html_result = await self.available_tools.execute(
                name="documentation",
                tool_input={
                    "command": "save_content",
                    "project_id": self.active_project_id,
                    "step_index": step_index,
                    "content": content,
                    "content_type": "html",
                    "title": title,
                    "metadata": {
                        "source": "browser_use"
                    }
                }
            )

            logger.info(f"HTML content saved for step {step_index}, result: {html_result}")
        except Exception as e:
            logger.error(f"Error saving HTML content: {str(e)}")
            logger.exception("Detailed exception:")

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return
        else:
            # Save any pending URLs before terminating
            if name.lower() == "terminate" and self.collected_urls and self.active_project_id:
                await self._save_collected_urls()

            await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
            await super()._handle_special_tool(name, result, **kwargs)

    async def get_browser_state(self) -> Optional[dict]:
        """Get the current browser state for context in next steps."""
        browser_tool = self.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool:
            return None

        try:
            # Get browser state directly from the tool with no context parameter
            result = await browser_tool.get_current_state()

            if result.error:
                logger.debug(f"Browser state error: {result.error}")
                return None

            # Store screenshot if available
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

            # Parse the state info
            return json.loads(result.output)

        except Exception as e:
            logger.debug(f"Failed to get browser state: {str(e)}")
            return None

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        # Store original prompt
        original_prompt = self.next_step_prompt

        # Only check recent messages (last 3) for browser activity
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            "browser_use" in msg.content.lower()
            for msg in recent_messages
            if hasattr(msg, "content") and isinstance(msg.content, str)
        )

        if browser_in_use:
            # Override with browser-specific prompt temporarily to get browser context
            self.next_step_prompt = BROWSER_NEXT_STEP_PROMPT

        # Call parent's think method
        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result
    # Documentation helper methods

    async def _init_documentation(self, request: str) -> None:
        """Initialize documentation for a new request"""
        try:
            # Initialize project
            logger.info(f"Calling documentation tool to initialize project: {self.active_project_id}")
            tool_input = {
                "command": "init_project",
                "project_id": self.active_project_id,
                "title": f"OpenManus Project: {request[:50]}..."
            }
            logger.info(f"Documentation tool input: {tool_input}")

            result = await self.available_tools.execute(
                name="documentation",
                tool_input=tool_input
            )

            logger.info(f"Documentation initialization result: {result}")
            logger.info(f"Documentation initialized with project ID: {self.active_project_id}")
        except Exception as e:
            logger.error(f"Error initializing documentation: {str(e)}")
            logger.exception("Detailed exception information:")

    async def _document_plan(self, plan_content: str) -> None:
        """Document the plan after it's created"""
        if not self.active_project_id:
            return

        try:
            await self.available_tools.execute(
                name="documentation",
                tool_input={
                    "command": "document_plan",
                    "project_id": self.active_project_id,
                    "content": plan_content,
                    "title": "Execution Plan"
                }
            )

            logger.info(f"Plan documented for project: {self.active_project_id}")
        except Exception as e:
            logger.error(f"Error documenting plan: {str(e)}")

    async def _document_step(self, step_index: int, step_content: str) -> None:
        """Document a step when it begins"""
        if not self.active_project_id:
            logger.warning("Skipping _document_step: no active project ID")
            return

        # If current_step_index is not None, store it in past_step_indices before updating
        if self.current_step_index is not None and self.current_step_index not in self.past_step_indices:
            logger.info(f"Saving previous step index {self.current_step_index} to history before moving to step {step_index}")
            self.past_step_indices.append(self.current_step_index)

        # Update the instance variable to keep track of the current step
        self.current_step_index = step_index
        logger.info(f"_document_step called with step_index={step_index}, set current_step_index={self.current_step_index}, past steps={self.past_step_indices}")

        try:
            step_result = await self.available_tools.execute(
                name="documentation",
                tool_input={
                    "command": "document_step",
                    "project_id": self.active_project_id,
                    "step_index": step_index,
                    "content": step_content,
                    "title": f"Step {step_index}: {step_content[:50]}..."
                }
            )

            logger.info(f"Step {step_index} documented for project: {self.active_project_id}, result: {step_result}")
        except Exception as e:
            logger.error(f"Error documenting step: {str(e)}")
            logger.exception("Detailed exception:")

    async def _document_action(self, tool_calls: List[ToolCall], result: str) -> None:
        """Document an action and its result (legacy method)"""
        # Get a step index, defaulting to 0 if needed
        step_index = self.current_step_index
        if step_index is None:
            logger.warning("current_step_index is None in _document_action, using 0 as fallback")
            step_index = 0

        # Call the new method with explicit step index
        await self._document_action_with_index(tool_calls, result, step_index)

    async def _save_html_content(self, content: str, title: str) -> None:
        """Save HTML content from browsing (legacy method)"""
        # Get a step index, defaulting to 0 if needed
        step_index = self.current_step_index
        if step_index is None:
            logger.warning("current_step_index is None in _save_html_content, using 0 as fallback")
            step_index = 0

        # Call the new method with explicit step index
        await self._save_html_content_with_index(content, title, step_index)

    async def _collect_browser_urls(self, tool_calls: List[ToolCall]) -> None:
        """Collect URLs from browser tool calls"""
        if not tool_calls:
            return

        for tool_call in tool_calls:
            if not tool_call or not tool_call.function:
                continue

            tool_name = tool_call.function.name

            if tool_name == "browser_use":
                try:
                    import json
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                    # Extract URL if present
                    url = args.get("url")
                    if url and url not in self.collected_urls:
                        self.collected_urls.append(url)

                except Exception:
                    pass

    async def _save_collected_urls(self) -> None:
        """Save collected URLs to documentation"""
        if not self.active_project_id or not self.collected_urls:
            return

        # Use a fallback step index if needed
        step_index = self.current_step_index
        if step_index is None:
            logger.warning("current_step_index is None, using 0 as fallback for URL documentation")
            step_index = 0

        logger.info(f"_save_collected_urls called with step_index={step_index}, URLs count={len(self.collected_urls)}")

        try:
            url_result = await self.available_tools.execute(
                name="documentation",
                tool_input={
                    "command": "save_urls",
                    "project_id": self.active_project_id,
                    "step_index": step_index,
                    "urls": self.collected_urls,
                    "title": "Browsing History"
                }
            )

            logger.info(f"Saved {len(self.collected_urls)} URLs for step {step_index}, result: {url_result}")
            self.collected_urls = []  # Reset after saving
        except Exception as e:
            logger.error(f"Error saving URLs: {str(e)}")
            logger.exception("Detailed exception:")

    async def _document_completion(self, final_result: str) -> None:
        """Document the completion of the request"""
        if not self.active_project_id:
            logger.warning("Skipping _document_completion: no active project ID")
            return

        # Use a high step index if none is set
        step_index = self.current_step_index if self.current_step_index is not None else 999
        logger.info(f"_document_completion called with step_index={step_index}")

        try:
            # Save final result as content
            completion_result = await self.available_tools.execute(
                name="documentation",
                tool_input={
                    "command": "save_content",
                    "project_id": self.active_project_id,
                    "step_index": step_index,
                    "content": final_result,
                    "content_type": "md",
                    "title": "Final Result",
                    "metadata": {
                        "final": True
                    }
                }
            )

            logger.info(f"Documented completion for project: {self.active_project_id}, result: {completion_result}")
        except Exception as e:
            logger.error(f"Error documenting completion: {str(e)}")
            logger.exception("Detailed exception:")
