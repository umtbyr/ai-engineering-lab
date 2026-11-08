import asyncio
import json
from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langchain.agents import create_agent


async def main():
    config = {
        "mcpServers": {
            "playwright": {
                "command": "cmd",
                "args": [
                    "/c",
                    "npx",
                    "--yes",
                    "@playwright/mcp@latest",
                    "--headless",
                    "--isolated",
                ],
            }
        }
    }

    async with MCPAdapter(config) as adapter:
        # 1. MCP Server'ın sunduğu araçları keşfet
        tools = await adapter.list_tools()

        print("=== DISCOVERED TOOLS ===")

        for tool in tools:
            print(f"- {tool.name}")

        # 2. browser_navigate aracını bul
        navigate = next(
            tool for tool in tools if tool.name.endswith("browser_navigate")
        )

        model = ChatOllama(
            model="qwen3:4b-instruct",
            temperature=0,
        )

        browser_tool_names = {
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
        }

        browser_tools = [tool for tool in tools if tool.name in browser_tool_names]

        print("Registered tools:", [tool.name for tool in browser_tools])

        agent = create_agent(
            model=model,
            tools=browser_tools,
            system_prompt=(
                "You are a browser research assistant. "
                "When the user explicitly asks to visit a website, "
                "use browser_navigate before answering. "
                "Use browser_snapshot to inspect page contents. "
                "Never claim to have visited a page without using a tool."
            ),
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Open https://docs.python.org/3/ " "using browser_navigate."
                        ),
                    }
                ]
            }
        )

        print("\n=== LANGGRAPH AGENT TEST ===")

        for message in result["messages"]:
            print(f"\n{type(message).__name__}")

            if hasattr(message, "tool_calls"):
                print("Tool calls:", message.tool_calls)

            if message.type == "tool":
                print("Tool result:", str(message.content)[:1000])


if __name__ == "__main__":
    asyncio.run(main())
