# OpenManus 项目研究报告

## 1. 项目概述

OpenManus 是一个开源的智能体（Agent）项目，旨在提供类似于 Manus 的功能，但无需邀请码即可使用。该项目由 MetaGPT 团队的成员在短时间内（3小时）构建完成，提供了一个简洁的实现方案，允许用户通过自然语言输入创意并由智能体执行相应的任务。

项目的主要特点：
- 支持多种工具调用（浏览器操作、Python代码执行、Google搜索等）
- 基于大语言模型（LLM）的智能交互
- 简单易用的命令行界面
- 可扩展的工具集合

## 2. 技术栈

### 2.1 编程语言与框架
- **Python 3.12**：项目的主要开发语言
- **Pydantic**：用于数据验证和设置管理
- **OpenAI API**：与大语言模型交互的主要接口
- **asyncio**：用于异步编程
- **tenacity**：用于重试机制

### 2.2 核心依赖
- **OpenAI Python SDK**：与OpenAI API交互
- **browser-use**：提供浏览器控制功能
- **toml**：配置文件解析

## 3. 代码结构

项目采用模块化设计，主要目录结构如下：

```
OpenManus/
├── app/                    # 主应用代码
│   ├── agent/              # 智能体实现
│   ├── flow/               # 流程控制
│   ├── prompt/             # 提示模板
│   ├── tool/               # 工具实现
│   ├── config.py           # 配置管理
│   ├── llm.py              # LLM交互
│   ├── logger.py           # 日志管理
│   └── schema.py           # 数据模型
├── config/                 # 配置文件
├── main.py                 # 主入口
└── run_flow.py             # 流程运行入口
```

### 3.1 主要模块说明

- **agent**：定义了不同类型的智能体，包括基础智能体、工具调用智能体和Manus智能体
- **tool**：实现了各种工具，如浏览器控制、Python代码执行、文件操作等
- **flow**：定义了智能体的执行流程
- **prompt**：存储与LLM交互的提示模板
- **schema**：定义了数据模型，如消息、工具调用等

## 4. 核心逻辑分析

### 4.1 智能体架构

项目采用了分层的智能体架构：

1. **BaseAgent**：基础智能体类，提供了智能体的基本功能
2. **ReActAgent**：实现了ReAct（Reasoning and Acting）模式的智能体
3. **ToolCallAgent**：扩展了ReAct智能体，增加了工具调用功能
4. **Manus**：最终用户使用的智能体，集成了多种工具

这种分层设计使得代码结构清晰，便于扩展和维护。

### 4.2 工具调用机制

工具调用是项目的核心功能，主要实现在 `app/agent/toolcall.py` 中。工具调用的流程如下：

1. **思考阶段（think）**：
   - 智能体将当前状态和用户输入发送给LLM
   - LLM返回思考结果和需要调用的工具
   - 智能体解析LLM的响应，提取工具调用信息

2. **行动阶段（act）**：
   - 智能体执行工具调用
   - 收集工具执行结果
   - 将结果添加到对话历史中

3. **工具执行（execute_tool）**：
   - 解析工具调用参数
   - 调用相应的工具
   - 处理工具执行结果
   - 处理特殊工具（如终止工具）

工具调用的核心代码：

```python
async def execute_tool(self, command: ToolCall) -> str:
    """Execute a single tool call with robust error handling"""
    if not command or not command.function or not command.function.name:
        return "Error: Invalid command format"

    name = command.function.name
    if name not in self.available_tools.tool_map:
        return f"Error: Unknown tool '{name}'"

    try:
        # Parse arguments
        args = json.loads(command.function.arguments or "{}")

        # Execute the tool
        logger.info(f"🔧 Activating tool: '{name}'...")
        result = await self.available_tools.execute(name=name, tool_input=args)

        # Format result for display
        observation = (
            f"Observed output of cmd `{name}` executed:\n{str(result)}"
            if result
            else f"Cmd `{name}` completed with no output"
        )

        # Handle special tools like `finish`
        await self._handle_special_tool(name=name, result=result)

        return observation
    except json.JSONDecodeError:
        error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
        logger.error(
            f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON"
        )
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
```

### 4.3 工具集合管理

项目使用 `ToolCollection` 类管理多个工具，提供了工具的注册、查找和执行功能：

```python
class ToolCollection:
    """A collection of defined tools."""

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)
```

### 4.4 LLM交互

项目使用 `LLM` 类与大语言模型交互，支持普通对话和工具调用两种模式：

1. **普通对话（ask）**：发送消息给LLM并获取文本响应
2. **工具调用（ask_tool）**：发送消息和工具定义给LLM，获取包含工具调用的响应

工具调用的核心代码：

```python
async def ask_tool(
    self,
    messages: List[Union[dict, Message]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    timeout: int = 60,
    tools: Optional[List[dict]] = None,
    tool_choice: Literal["none", "auto", "required"] = "auto",
    temperature: Optional[float] = None,
    **kwargs,
):
    # ...省略部分代码...
    
    # Set up the completion request
    response = await self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=temperature or self.temperature,
        max_tokens=self.max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        timeout=timeout,
        **kwargs,
    )

    # Check if response is valid
    if not response.choices or not response.choices[0].message:
        print(response)
        raise ValueError("Invalid or empty response from LLM")

    return response.choices[0].message
```

### 4.5 浏览器控制

项目使用 `BrowserUseTool` 类实现浏览器控制功能，支持多种浏览器操作：

- 导航（navigate）
- 点击元素（click）
- 输入文本（input_text）
- 截图（screenshot）
- 获取HTML（get_html）
- 执行JavaScript（execute_js）
- 滚动页面（scroll）
- 标签页管理（switch_tab、new_tab、close_tab）
- 刷新页面（refresh）

浏览器工具基于 `browser-use` 库实现，提供了丰富的浏览器交互功能。

## 5. 总结与评价

OpenManus 是一个设计良好的智能体项目，具有以下特点：

1. **模块化设计**：项目结构清晰，各模块职责明确，便于扩展和维护
2. **强大的工具调用**：支持多种工具，能够完成复杂的任务
3. **灵活的配置**：通过配置文件可以轻松调整项目行为
4. **良好的错误处理**：对各种异常情况进行了处理，提高了系统的稳定性

项目的核心价值在于提供了一个开源的、可扩展的智能体框架，使用户能够通过自然语言与智能体交互，并利用智能体完成各种任务。

未来可能的改进方向：
- 增强规划能力
- 添加更多工具支持
- 实现实时演示和回放功能
- 使用强化学习微调模型
- 建立全面的性能基准测试 