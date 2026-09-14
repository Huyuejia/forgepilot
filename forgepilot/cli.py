"""命令行入口。

这个模块负责把“用户怎么启动 ForgePilot”翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。
"""

import argparse  # 导入参数解析模块，用于处理命令行输入的参数（如 --model）
import os        # 导入操作系统接口模块，用于读取环境变量
import shutil    # 导入高级文件操作模块，这里主要用于获取终端窗口的大小
import sys       # 导入系统相关模块，用于处理退出码和标准错误流输出
import textwrap  # 导入文本包装模块，用于处理多行字符串的缩进和格式化

from .config import load_project_env, provider_env  # 导入加载项目环境变量和获取服务商环境配置的工具
from .models import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient  # 导入不同 AI 厂商的客户端类
from .runtime import ForgePilot, SessionStore  # 导入代理运行时的核心类和会话存储管理类
from .workspace import WorkspaceContext, middle  # 导入工作区上下文管理工具和字符串截断工具

# 定义一个元组，存储默认需要脱敏的敏感环境变量名称（防止在日志中泄露 API Key）
DEFAULT_SECRET_ENV_NAMES = (
    "FORGEPILOT_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "FORGEPILOT_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "FORGEPILOT_DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY",
    "FORGEPILOT_RIGHT_CODES_API_KEY",
    "RIGHT_CODES_API_KEY",
    "GITHUB_PAT",
    "GH_PAT",
)

# 欢迎界面使用的 ASCII 艺术字符画（一只小猫）
WELCOME_ART = (
    "        /\\___/\\\\",
    "       (  o o  )",
    "       /   ^   \\\\",
    "      /|       |\\\\",
)
WELCOME_NAME = "ForgePilot"  # 代理名称
WELCOME_SUBTITLE = "local coding agent"  # 代理副标题
WELCOME_STATUS = "calm shell, ready for work"  # 状态描述
HELP_DETAILS = textwrap.dedent(  # 使用 textwrap.dedent 自动去除多行字符串前面的缩进，保持格式整洁
    """\
    Commands:
    /help    Show this help message.
    /memory  Show the agent's distilled working memory.
    /plan    Show the current task plan.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()


DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"  # 默认的本地 Ollama 模型名称
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"  # 默认的 Ollama 服务器地址
DEFAULT_OPENAI_MODEL = "gpt-5.4"  # 默认的 OpenAI 兼容模型名称
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"  # 默认的 OpenAI 接口地址
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"  # 默认的 Anthropic 兼容模型名称
DEFAULT_ANTHROPIC_BASE_URL = "https://www.right.codes/claude/v1"  # 默认的 Anthropic 接口地址
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"  # 默认的 DeepSeek 模型名称
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"  # 默认的 DeepSeek 接口地址
LEGACY_SECRET_ENV_NAMES_VAR = "MINI_CODING_AGENT_SECRET_ENV_NAMES"  # 旧版的敏感变量配置名
SECRET_ENV_NAMES_VAR = "FORGEPILOT_SECRET_ENV_NAMES"  # 新版的敏感变量配置名

# 定义一个内部函数，用于确定最终使用的模型名称
def _effective_model(args, provider):
    # getattr 用于从 args 对象中安全地获取 'model' 属性，如果不存在则返回 None
    explicit_model = getattr(args, "model", None) 
    if explicit_model:
        return explicit_model  # 优先级 1: 如果命令行指定了模型，直接使用
    if provider == "openai":
        # 优先级 2: 从环境变量读取，支持新旧两种环境变量名
        model = provider_env("FORGEPILOT_OPENAI_MODEL", ("OPENAI_MODEL",))
        if model:
            return model
        return DEFAULT_OPENAI_MODEL  # 优先级 3: 使用代码中定义的默认值
    if provider == "anthropic":
        model = provider_env("FORGEPILOT_ANTHROPIC_MODEL", ("ANTHROPIC_MODEL",))
        if model:
            return model
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "deepseek":
        model = provider_env("FORGEPILOT_DEEPSEEK_MODEL", ("DEEPSEEK_MODEL",))
        if model:
            return model
        return DEFAULT_DEEPSEEK_MODEL
    return DEFAULT_OLLAMA_MODEL  # 默认返回 Ollama 的配置

# 获取所有需要脱敏处理的环境变量名列表
def _configured_secret_names(args):
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)  # 使用 set 集合去重
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)  # 添加命令行传入的敏感变量名
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")  # 从系统环境变量中读取额外的敏感列表
    if not extra_names.strip():
        extra_names = os.environ.get(LEGACY_SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")  # 按逗号分隔多个变量名
            if item.strip()
        )
    return sorted(configured_secret_names)  # 返回排序后的列表

# 根据命令行参数构建对应的 AI 模型客户端实例
def _build_model_client(args):
    provider = getattr(args, "provider", "openai")  # 获取服务商类型，默认为 openai
    # CLI 只负责把 provider 选择翻译成具体 client。
    # 真正的提示词格式、缓存支持、HTTP 协议差异，都封装在 models.py 里。
    if provider == "openai":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("FORGEPILOT_OPENAI_API_BASE", ("OPENAI_API_BASE",), DEFAULT_OPENAI_BASE_URL)  # 确定 API 地址
        api_key = provider_env("FORGEPILOT_OPENAI_API_KEY", ("OPENAI_API_KEY",))  # 确定 API Key
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )  # 返回一个 OpenAI 兼容客户端实例
    if provider == "anthropic":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("FORGEPILOT_ANTHROPIC_API_BASE", ("ANTHROPIC_API_BASE",), DEFAULT_ANTHROPIC_BASE_URL)
        api_key = provider_env(
            "FORGEPILOT_ANTHROPIC_API_KEY",
            ("ANTHROPIC_API_KEY", "FORGEPILOT_RIGHT_CODES_API_KEY", "RIGHT_CODES_API_KEY", "FORGEPILOT_OPENAI_API_KEY", "OPENAI_API_KEY"),
        )
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "deepseek":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("FORGEPILOT_DEEPSEEK_API_BASE", ("DEEPSEEK_API_BASE",), DEFAULT_DEEPSEEK_BASE_URL)
        api_key = provider_env("FORGEPILOT_DEEPSEEK_API_KEY", ("DEEPSEEK_API_KEY",))
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )

    model = _effective_model(args, provider)  # 如果是本地 Ollama
    host = getattr(args, "host", DEFAULT_OLLAMA_HOST)  # 获取本地主机地址
    return OllamaModelClient(  # 返回本地模型客户端实例
        model=model,
        host=host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
    )

# 构建欢迎界面的 UI (TUI) 字符串
def build_welcome(agent, model, host):
    # 动态计算宽度，根据当前终端窗口大小在 68 到 84 字符之间调整
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4  # 内部可用宽度
    gap = 3  # 两栏布局之间的间距
    left_width = (inner - gap) // 2  # 左栏宽度
    right_width = inner - gap - left_width  # 右栏宽度

    def row(text):  # 定义一个内部辅助函数，生成一行两边带竖线的文本
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):  # 生成分隔线
        return "+" + char * (width - 2) + "+"

    def center(text):  # 生成居中文本行
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):  # 生成带标签的单元格数据
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):  # 生成双栏展示的一行
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")  # 最外层的双线
    rows = [center(text) for text in WELCOME_ART]  # 添加 ASCII 艺术
    rows.extend(
        [
            center(WELCOME_NAME),  # 添加名称
            center(WELCOME_SUBTITLE),
            center(WELCOME_STATUS),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),  # 显示当前工作目录
            pair("MODEL", model, "BRANCH", agent.workspace.branch),  # 显示模型和分支
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),  # 显示审批策略和会话 ID
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])  # 拼接所有行并返回

# 构建 Agent 实例，这是 CLI 到 Runtime 的转换核心
def build_agent(args):
    workspace = WorkspaceContext.build(args.cwd)  # 1. 扫描当前目录，建立工作区快照
    load_project_env(workspace.repo_root)  # 2. 尝试加载项目根目录下的 .env 文件
    configured_secret_names = _configured_secret_names(args)  # 3. 确定脱敏名单
    store = SessionStore(workspace.repo_root + "/.ForgePilot/sessions")  # 4. 初始化会话存储器
    model = _build_model_client(args)  # 5. 构建 AI 客户端
    session_id = args.resume  # 6. 处理恢复会话逻辑
    if session_id == "latest":
        session_id = store.latest()  # 如果参数是 latest，找最近的一个 json 文件
    if session_id:
        return ForgePilot.from_session(  # 从旧的会话数据恢复 ForgePilot 实例
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=configured_secret_names,
        )
    return ForgePilot(  # 否则，创建一个全新的 ForgePilot 实例
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names,
    )

# 定义命令行参数解析器
def build_arg_parser():
    parser = argparse.ArgumentParser(  # 创建解析对象
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # 自动在帮助文档里显示默认值
        description="ForgePilot local coding agent for Ollama, OpenAI-compatible, Anthropic-compatible, or DeepSeek models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")  # 位置参数：任务描述
    parser.add_argument("--cwd", default=".", help="Workspace directory.")  # 参数：指定工作目录
    parser.add_argument("--provider", choices=("ollama", "openai", "anthropic", "deepseek"), default="openai", help="Model backend to use.")  # 参数：选择厂商
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to qwen3.5:4b for Ollama, FORGEPILOT_OPENAI_MODEL for openai, FORGEPILOT_ANTHROPIC_MODEL for anthropic, and FORGEPILOT_DEEPSEEK_MODEL for deepseek when set.",
    )
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL.")
    parser.add_argument("--base-url", default=None, help="Provider API base URL for openai, anthropic, or deepseek.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")  # 模型随机度
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")  # 核采样阈值
    return parser  # 返回配置好的解析器

# 程序执行的主入口函数
def main(argv=None):
    args = build_arg_parser().parse_args(argv)  # 1. 解析传入的命令行参数
    agent = build_agent(args)  # 2. 根据参数构造代理对象

    # 获取模型和主机信息用于 UI 显示
    model = getattr(agent.model_client, "model", getattr(args, "model", DEFAULT_OLLAMA_MODEL)) 
    host = getattr(agent.model_client, "host", getattr(agent.model_client, "base_url", getattr(args, "host", DEFAULT_OLLAMA_HOST)))
    print(build_welcome(agent, model=model, host=host))  # 打印欢迎面板

    if args.prompt:
        # one-shot 模式：只跑一次 ask，不进入 REPL 循环。
        prompt = " ".join(args.prompt).strip()  # 把所有参数拼成一段文字
        if prompt:
            print()
            try:
                print(agent.ask(prompt))  # 执行任务并打印结果
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)  # 如果出错，打印到标准错误流
                return 1
        return 0

    while True:  # REPL 模式：无限循环等待用户输入
        # 交互模式：每次读取一条用户输入，交给同一个 agent，
        # 因此 session history 和 working memory 会跨轮延续。
        try:
            user_input = input("\nforgepilot> ").strip()  # 获取用户输入
        except (EOFError, KeyboardInterrupt):  # 捕获 Ctrl+C 或 Ctrl+D
            print("")
            return 0  # 优雅退出

        if not user_input:  # 忽略空行
            continue
        if user_input in {"/exit", "/quit"}:  # 特殊退出命令
            return 0
        if user_input == "/help":  # 显示帮助
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":  # 查看记忆
            print(agent.memory_text())
            continue
        if user_input == "/plan":  # 查看当前计划
            print(agent.plan_text())
            continue
        if user_input == "/session":  # 查看会话文件路径
            print(agent.session_path)
            continue
        if user_input == "/reset":  # 重置会话
            agent.reset()
            print("session reset")
            continue

        print()  # 打印空行美化输出
        try:
            print(agent.ask(user_input))  # 核心：调用 agent 执行任务
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
