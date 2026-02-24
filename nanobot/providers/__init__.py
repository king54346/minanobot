"""nanobot.providers — LLM 提供商抽象层。

对外导出：
- LLMProvider / LLMResponse — 基类与标准返回结构（必选）
- LiteLLMProvider           — 基于 litellm 的默认实现（必选）
- LangChainProvider         — 基于 langchain-openai 的可选实现
                              （需额外安装: pip install nanobot-ai[langchain]）
"""

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.litellm_provider import LiteLLMProvider

# LangChainProvider 是可选依赖，仅在 langchain-openai 已安装时才可用。
# 采用惰性导入，避免未装 langchain 时让整个 providers 包不可用。
try:
    from nanobot.providers.langchain_provider import LangChainProvider
except ImportError:
    LangChainProvider = None  # type: ignore[assignment,misc]

# 公共导出：上层通常只需要这些 symbol。
__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "LangChainProvider"]
