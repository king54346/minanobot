"""
nanobot.providers.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

Provider Registry — LLM 提供商元数据的 **唯一权威来源**。

设计目标
--------
整个项目中所有与"提供商"相关的行为 —— 环境变量注入、模型名前缀处理、
配置匹配、``nanobot status`` 状态展示 —— 全部从本文件中的 ``PROVIDERS``
元组派生，而不是散落在各处硬编码。这保证了：
  - 新增提供商只需改一处（本文件 + config/schema.py 各加一条）；
  - 所有逻辑自动同步，不会遗漏。

核心概念
--------
1. **ProviderSpec（提供商规格）**
   一个不可变的 dataclass，描述单个提供商的所有元数据：
   名称、关键词、环境变量、LiteLLM 前缀、网关检测规则等。

2. **PROVIDERS（注册表）**
   一个有序元组，包含所有已注册的 ProviderSpec。
   **顺序 = 优先级**：网关（Gateway）排在前面，标准提供商居中，
   辅助提供商（如 Groq）排在最后。匹配和回退都遵循此顺序。

3. **Lookup 辅助函数**
   - ``find_by_model(model)``   — 根据模型名关键词匹配标准提供商
   - ``find_gateway(...)``      — 根据 api_key 前缀或 api_base URL 检测网关
   - ``find_by_name(name)``     — 根据配置字段名精确查找

新增提供商步骤
--------------
  1. 在下方 ``PROVIDERS`` 元组中添加一条 ``ProviderSpec``。
  2. 在 ``config/schema.py`` 的 ``ProvidersConfig`` 中添加对应字段。
  完成。环境变量、前缀、配置匹配、状态展示全部自动派生。

提供商分类
----------
- **Gateway（网关）**: OpenRouter、AiHubMix 等，可路由任意模型，
  通过 api_key 前缀或 api_base URL 关键词检测。
- **Standard（标准）**: Anthropic、OpenAI、DeepSeek 等，
  通过模型名中的关键词匹配（如 "claude" → Anthropic）。
- **Local（本地）**: vLLM 等本地部署，通过配置键名显式指定。
- **Auxiliary（辅助）**: Groq 等，主要用于语音转写，也可用作 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """单个 LLM 提供商的元数据规格。

    每个 ProviderSpec 实例描述一个提供商的完整信息，包括：
    - 如何识别它（名称、关键词、api_key 前缀等）
    - 如何与 LiteLLM 交互（模型名前缀、环境变量）
    - 如何在 UI 中展示（display_name）

    Placeholders（占位符）
    ~~~~~~~~~~~~~~~~~~~~~~
    ``env_extras`` 的值中可使用以下占位符，运行时会被替换：
      - ``{api_key}``  — 用户配置的 API Key
      - ``{api_base}`` — 用户配置的 api_base，或本 spec 的 default_api_base

    示例参见下方 PROVIDERS 元组中的真实条目。
    """

    # ======================== 身份标识 ========================
    name: str                       # 配置字段名，如 "dashscope"，对应 config.json 中 providers.dashscope
    keywords: tuple[str, ...]       # 模型名关键词（小写），用于从模型名推断提供商，如 ("qwen", "dashscope")
    env_key: str                    # 对应的 LiteLLM 环境变量名，如 "DASHSCOPE_API_KEY"
    display_name: str = ""          # 在 `nanobot status` 中展示的友好名称，为空时用 name.title()

    # ======================== 模型名前缀 ========================
    # LiteLLM 需要特定前缀来路由到正确的提供商。
    # 例如 dashscope 的模型 "qwen-max" 需要变成 "dashscope/qwen-max"。
    litellm_prefix: str = ""                 # LiteLLM 前缀，如 "dashscope" → 模型变为 "dashscope/{model}"
    skip_prefixes: tuple[str, ...] = ()      # 如果模型名已经以这些前缀开头，则跳过添加前缀（避免重复）

    # ======================== 额外环境变量 ========================
    # 某些提供商需要设置多个环境变量。
    # 格式：(("ENV_VAR_NAME", "value_template"), ...)
    # 值模板中可使用 {api_key} 和 {api_base} 占位符。
    # 例如智谱需要同时设置 ZAI_API_KEY 和 ZHIPUAI_API_KEY。
    env_extras: tuple[tuple[str, str], ...] = ()

    # ======================== 网关 / 本地检测 ========================
    is_gateway: bool = False                 # 是否为网关（如 OpenRouter），网关可以路由任意模型
    is_local: bool = False                   # 是否为本地部署（如 vLLM、Ollama）
    detect_by_key_prefix: str = ""           # 通过 api_key 前缀自动检测，如 "sk-or-" → OpenRouter
    detect_by_base_keyword: str = ""         # 通过 api_base URL 中的关键词检测，如 "aihubmix"
    default_api_base: str = ""               # 默认的 API 端点 URL（用户未配置 api_base 时的回退值）

    # ======================== 网关行为 ========================
    # 某些网关（如 AiHubMix）不理解 "anthropic/claude-3" 这样的带前缀模型名，
    # 需要先剥离前缀再重新添加网关自己的前缀。
    strip_model_prefix: bool = False         # 是否先剥离 "provider/" 前缀再重新添加 litellm_prefix

    # ======================== 模型级参数覆盖 ========================
    # 某些特定模型需要强制覆盖参数。
    # 例如 Kimi K2.5 的 API 要求 temperature >= 1.0。
    # 格式：(("model_name", {"param": value}), ...)
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    @property
    def label(self) -> str:
        """用于展示的提供商标签名。优先使用 display_name，否则用 name 的首字母大写形式。"""
        return self.display_name or self.name.title()


# ---------------------------------------------------------------------------
# PROVIDERS — 提供商注册表（有序元组）
#
# ⚠️ 顺序 = 优先级：匹配和回退时从上往下遍历，先匹配到的优先。
#    排列规则：网关 > 标准提供商 > 本地部署 > 辅助提供商
#
# 新增提供商时，可以复制任意一条作为模板，填入所有字段即可。
# ---------------------------------------------------------------------------

PROVIDERS: tuple[ProviderSpec, ...] = (

    # ===================================================================
    # Custom — 用户自定义的 OpenAI 兼容端点
    # ===================================================================
    # 没有自动检测机制，仅当用户在 config.json 中显式配置 "custom" 时激活。
    # 适用于：用户自建的 API 网关、内部代理等。

    ProviderSpec(
        name="custom",
        keywords=(),
        env_key="OPENAI_API_KEY",
        display_name="Custom",
        litellm_prefix="openai",
        skip_prefixes=("openai/",),
        is_gateway=True,
        strip_model_prefix=True,
    ),

    # ===================================================================
    # Gateway（网关） — 通过 api_key 前缀或 api_base URL 检测，而非模型名
    # ===================================================================
    # 网关可以路由任意模型，因此在回退匹配中优先级最高。

    # OpenRouter：全球 LLM 网关，API Key 以 "sk-or-" 开头。
    # 支持 100+ 模型（OpenAI / Anthropic / Google 等），统一计费。
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        env_key="OPENROUTER_API_KEY",
        display_name="OpenRouter",
        litellm_prefix="openrouter",        # claude-3 → openrouter/claude-3
        skip_prefixes=(),
        env_extras=(),
        is_gateway=True,
        is_local=False,
        detect_by_key_prefix="sk-or-",
        detect_by_base_keyword="openrouter",
        default_api_base="https://openrouter.ai/api/v1",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # AiHubMix：全球 LLM 网关，OpenAI 兼容接口。
    # strip_model_prefix=True：AiHubMix 不理解 "anthropic/claude-3" 这样的带前缀名，
    # 所以需要先剥离前缀到裸名 "claude-3"，再重新添加为 "openai/claude-3"。
    ProviderSpec(
        name="aihubmix",
        keywords=("aihubmix",),
        env_key="OPENAI_API_KEY",           # OpenAI-compatible
        display_name="AiHubMix",
        litellm_prefix="openai",            # → openai/{model}
        skip_prefixes=(),
        env_extras=(),
        is_gateway=True,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="aihubmix",
        default_api_base="https://aihubmix.com/v1",
        strip_model_prefix=True,            # anthropic/claude-3 → claude-3 → openai/claude-3
        model_overrides=(),
    ),

    # ===================================================================
    # Standard（标准提供商） — 通过模型名关键词匹配
    # ===================================================================

    # Anthropic：LiteLLM 原生识别 "claude-*" 系列模型名，无需添加前缀。
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        litellm_prefix="",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # OpenAI：LiteLLM 原生识别 "gpt-*" 系列模型名，无需添加前缀。
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt"),
        env_key="OPENAI_API_KEY",
        display_name="OpenAI",
        litellm_prefix="",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # DeepSeek：需要 "deepseek/" 前缀才能让 LiteLLM 正确路由。
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        env_key="DEEPSEEK_API_KEY",
        display_name="DeepSeek",
        litellm_prefix="deepseek",          # deepseek-chat → deepseek/deepseek-chat
        skip_prefixes=("deepseek/",),       # avoid double-prefix
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # Gemini：需要 "gemini/" 前缀才能让 LiteLLM 正确路由。
    ProviderSpec(
        name="gemini",
        keywords=("gemini",),
        env_key="GEMINI_API_KEY",
        display_name="Gemini",
        litellm_prefix="gemini",            # gemini-pro → gemini/gemini-pro
        skip_prefixes=("gemini/",),         # avoid double-prefix
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # 智谱 AI（Zhipu）：LiteLLM 使用 "zai/" 前缀路由。
    # 额外镜像 API Key 到 ZHIPUAI_API_KEY（LiteLLM 部分代码路径会检查该变量）。
    # skip_prefixes：如果模型名已经被网关前缀过（如 "openrouter/..."），则不再添加 "zai/"。
    ProviderSpec(
        name="zhipu",
        keywords=("zhipu", "glm", "zai"),
        env_key="ZAI_API_KEY",
        display_name="Zhipu AI",
        litellm_prefix="zai",              # glm-4 → zai/glm-4
        skip_prefixes=("zhipu/", "zai/", "openrouter/", "hosted_vllm/"),
        env_extras=(
            ("ZHIPUAI_API_KEY", "{api_key}"),
        ),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # 阿里云 DashScope（通义千问）：Qwen 系列模型，需要 "dashscope/" 前缀。
    ProviderSpec(
        name="dashscope",
        keywords=("qwen", "dashscope"),
        env_key="DASHSCOPE_API_KEY",
        display_name="DashScope",
        litellm_prefix="dashscope",         # qwen-max → dashscope/qwen-max
        skip_prefixes=("dashscope/", "openrouter/"),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # Moonshot（月之暗面）：Kimi 系列模型，需要 "moonshot/" 前缀。
    # LiteLLM 需要通过 MOONSHOT_API_BASE 环境变量找到端点。
    # Kimi K2.5 的 API 强制要求 temperature >= 1.0。
    ProviderSpec(
        name="moonshot",
        keywords=("moonshot", "kimi"),
        env_key="MOONSHOT_API_KEY",
        display_name="Moonshot",
        litellm_prefix="moonshot",          # kimi-k2.5 → moonshot/kimi-k2.5
        skip_prefixes=("moonshot/", "openrouter/"),
        env_extras=(
            ("MOONSHOT_API_BASE", "{api_base}"),
        ),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="https://api.moonshot.ai/v1",   # intl; use api.moonshot.cn for China
        strip_model_prefix=False,
        model_overrides=(
            ("kimi-k2.5", {"temperature": 1.0}),
        ),
    ),

    # MiniMax：需要 "minimax/" 前缀才能让 LiteLLM 正确路由。
    # 使用 OpenAI 兼容 API，端点为 api.minimax.io/v1。
    ProviderSpec(
        name="minimax",
        keywords=("minimax",),
        env_key="MINIMAX_API_KEY",
        display_name="MiniMax",
        litellm_prefix="minimax",            # MiniMax-M2.1 → minimax/MiniMax-M2.1
        skip_prefixes=("minimax/", "openrouter/"),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="https://api.minimax.io/v1",
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # ===================================================================
    # Local（本地部署） — 通过配置键名匹配，而非 api_base URL
    # ===================================================================

    # vLLM / 任意 OpenAI 兼容的本地服务器。
    # 当 config.json 中的配置键为 "vllm"（provider_name="vllm"）时激活。
    # 用户必须在配置中提供 api_base。
    ProviderSpec(
        name="vllm",
        keywords=("vllm",),
        env_key="HOSTED_VLLM_API_KEY",
        display_name="vLLM/Local",
        litellm_prefix="hosted_vllm",      # Llama-3-8B → hosted_vllm/Llama-3-8B
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=True,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",                # user must provide in config
        strip_model_prefix=False,
        model_overrides=(),
    ),

    # ===================================================================
    # Auxiliary（辅助提供商） — 非主要 LLM 提供商
    # ===================================================================

    # Groq：主要用于 Whisper 语音转写，也可以作为 LLM 使用。
    # 需要 "groq/" 前缀才能让 LiteLLM 正确路由。
    # 排在最后 —— 在回退匹配中它很少会被命中。
    ProviderSpec(
        name="groq",
        keywords=("groq",),
        env_key="GROQ_API_KEY",
        display_name="Groq",
        litellm_prefix="groq",              # llama3-8b-8192 → groq/llama3-8b-8192
        skip_prefixes=("groq/",),           # avoid double-prefix
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
    ),
)


# ---------------------------------------------------------------------------
# Lookup 辅助函数 — 根据不同维度查找匹配的 ProviderSpec
# ---------------------------------------------------------------------------

def find_by_model(model: str) -> ProviderSpec | None:
    """根据模型名关键词匹配标准提供商（大小写不敏感）。

    遍历 PROVIDERS 元组，跳过网关（Gateway）和本地部署（Local），
    因为它们不通过模型名匹配，而是通过 api_key 前缀或 api_base URL 检测。

    Args:
        model: 模型名，如 "gpt-4o"、"claude-3-sonnet"、"qwen-max" 等。

    Returns:
        匹配到的 ProviderSpec，未找到则返回 None。

    示例::

        find_by_model("claude-3-sonnet")  # → Anthropic 的 ProviderSpec
        find_by_model("qwen-max")         # → DashScope 的 ProviderSpec
        find_by_model("unknown-model")    # → None
    """
    model_lower = model.lower()
    for spec in PROVIDERS:
        if spec.is_gateway or spec.is_local:
            continue
        if any(kw in model_lower for kw in spec.keywords):
            return spec
    return None


def find_gateway(
    provider_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    """检测网关（Gateway）或本地部署（Local）提供商。

    检测优先级：
      1. **provider_name** — 如果直接映射到某个网关/本地 spec，则立即使用。
      2. **api_key 前缀** — 如 "sk-or-" 开头 → OpenRouter。
      3. **api_base URL 关键词** — 如 URL 中包含 "aihubmix" → AiHubMix。

    设计注意事项：
      带有自定义 api_base 的标准提供商（如 DeepSeek 经过代理访问）
      **不会**被误判为 vLLM —— 旧版的兜底逻辑已移除。

    Args:
        provider_name: 配置中的提供商键名，如 "openrouter"、"vllm"。
        api_key: 用户的 API Key，用于前缀检测。
        api_base: 用户的 API 端点 URL，用于关键词检测。

    Returns:
        匹配到的网关/本地 ProviderSpec，未找到则返回 None。
    """
    # 1. 通过配置键名直接匹配
    if provider_name:
        spec = find_by_name(provider_name)
        if spec and (spec.is_gateway or spec.is_local):
            return spec

    # 2. 自动检测：api_key 前缀 / api_base URL 关键词
    for spec in PROVIDERS:
        if spec.detect_by_key_prefix and api_key and api_key.startswith(spec.detect_by_key_prefix):
            return spec
        if spec.detect_by_base_keyword and api_base and spec.detect_by_base_keyword in api_base:
            return spec

    return None


def find_by_name(name: str) -> ProviderSpec | None:
    """根据配置字段名精确查找提供商规格。

    Args:
        name: 配置字段名，如 "dashscope"、"openrouter"、"vllm" 等。

    Returns:
        匹配到的 ProviderSpec，未找到则返回 None。
    """
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None
