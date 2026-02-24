"""
skill加载器：为 Agent 加载能力扩展
skill来源（按优先级排序）：
1. 工作空间技能：workspace/skills/<技能名>/SKILL.md（用户自定义，优先级最高）
2. 内置技能：nanobot/skills/<技能名>/SKILL.md（项目自带）

技能加载方式：
- "始终加载"（always=true）：每次对话都注入到上下文中
- "按需加载"（渐进式）：先给 Agent 一个技能摘要，Agent 需要时再用 read_file 读取完整内容
"""

import json
import os
import re
import shutil
from pathlib import Path

# 内置技能目录（相对于当前文件的路径：nanobot/skills/）
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillsLoader:
    """
    技能加载器

    负责发现、加载和管理 Agent 的技能文件。

    技能目录结构示例：
        skills/
            weather/
                SKILL.md      ← 技能描述文件（教 Agent 如何查天气）
            github/
                SKILL.md      ← 技能描述文件（教 Agent 如何操作 GitHub）

    技能文件支持 YAML frontmatter 元数据：
        ---
        description: "查询天气信息"
        always: true              ← 是否每次对话都加载
        metadata: '{"nanobot": {"requires": {"bins": ["curl"], "env": ["API_KEY"]}}}'
        ---
        # 技能正文内容...
    """
    
    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        """
        初始化技能加载器

        Args:
            workspace: 工作空间路径
            builtin_skills_dir: 内置技能目录路径，默认为 nanobot/skills/
        """
        self.workspace = workspace                              # 工作空间路径
        self.workspace_skills = workspace / "skills"            # 用户自定义技能目录
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR  # 内置技能目录

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        列出所有可用的技能

        扫描工作空间技能目录和内置技能目录，查找所有包含 SKILL.md 的子目录。
        工作空间技能优先级高于内置技能（同名时工作空间的覆盖内置的）。

        Args:
            filter_unavailable: 如果为 True，过滤掉不满足依赖要求的技能
                               （例如缺少必需的命令行工具或环境变量）

        Returns:
            技能信息字典列表，每个字典包含：
            - name: 技能名称（目录名）
            - path: SKILL.md 的完整路径
            - source: 来源（"workspace" 或 "builtin"）
        """
        skills = []
        
        # 第一优先级：工作空间技能（用户自定义）
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})
        
        # 第二优先级：内置技能（同名的不会重复添加，工作空间的优先）
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    # 只有不存在同名工作空间技能时，才添加内置技能
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})
        
        # 按依赖要求过滤
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_meta(s["name"]))]
        return skills
    
    def load_skill(self, name: str) -> str | None:
        """
        按名称加载单个技能的内容

        查找顺序：工作空间技能 → 内置技能

        Args:
            name: 技能名称（即目录名，如 "weather"）

        Returns:
            技能文件的完整文本内容，未找到则返回 None
        """
        # 优先查找工作空间技能
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")
        
        # 再查找内置技能
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return builtin_skill.read_text(encoding="utf-8")
        
        return None
    
    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        加载指定技能并格式化为 Agent 上下文内容

        将多个技能的正文内容拼接成一段格式化文本，
        注入到 Agent 的系统提示词中，让 Agent 获得这些"操作手册"。

        注意：会去除 frontmatter 元数据，只保留技能正文。

        Args:
            skill_names: 要加载的技能名称列表

        Returns:
            格式化后的技能内容文本，多个技能之间用 --- 分隔
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)  # 去除 YAML frontmatter
                parts.append(f"### Skill: {name}\n\n{content}")
        
        return "\n\n---\n\n".join(parts) if parts else ""
    
    def build_skills_summary(self) -> str:
        """
        构建所有技能的摘要信息（XML 格式）

        这是"渐进式加载"的关键：
        - 不直接加载全部技能内容（避免上下文过大）
        - 只给 Agent 一个摘要列表，包含名称、描述、文件路径
        - Agent 需要某个技能时，自己用 read_file 工具去读取完整内容

        生成的 XML 格式示例：
            <skills>
              <skill available="true">
                <name>weather</name>
                <description>查询天气信息</description>
                <location>/path/to/SKILL.md</location>
              </skill>
              <skill available="false">
                <name>github</name>
                <description>GitHub 操作</description>
                <location>/path/to/SKILL.md</location>
                <requires>CLI: gh, ENV: GITHUB_TOKEN</requires>
              </skill>
            </skills>

        Returns:
            XML 格式的技能摘要，无技能时返回空字符串
        """
        # 获取所有技能（包括不可用的，因为要在摘要中标注 available 状态）
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""
        
        # XML 特殊字符转义
        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            skill_meta = self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)  # 检查依赖是否满足

            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")
            
            # 如果技能不可用，显示缺失的依赖信息
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")
            
            lines.append(f"  </skill>")
        lines.append("</skills>")
        
        return "\n".join(lines)
    
    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """
        获取技能缺失的依赖信息

        检查技能需要的命令行工具（bins）和环境变量（env），
        返回所有缺失项的描述字符串。

        Args:
            skill_meta: 技能的 nanobot 元数据

        Returns:
            缺失依赖的描述（如 "CLI: gh, ENV: GITHUB_TOKEN"），无缺失则返回空字符串
        """
        missing = []
        requires = skill_meta.get("requires", {})
        # 检查命令行工具是否存在
        for b in requires.get("bins", []):
            if not shutil.which(b):          # which 查找命令是否在 PATH 中
                missing.append(f"CLI: {b}")
        # 检查环境变量是否设置
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)
    
    def _get_skill_description(self, name: str) -> str:
        """
        获取技能的描述文本

        从技能的 frontmatter 中读取 description 字段，
        如果没有设置，则回退使用技能名称。

        Args:
            name: 技能名称

        Returns:
            技能描述字符串
        """
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # 回退：用技能名称作为描述

    def _strip_frontmatter(self, content: str) -> str:
        """
        去除 Markdown 内容中的 YAML frontmatter 头部

        frontmatter 格式：
            ---
            description: "xxx"
            always: true
            ---
            正文内容...

        去除后只保留正文内容。

        Args:
            content: 包含 frontmatter 的 Markdown 内容

        Returns:
            去除 frontmatter 后的纯正文内容
        """
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content
    
    def _parse_nanobot_metadata(self, raw: str) -> dict:
        """
        解析 frontmatter 中的 nanobot 专用元数据

        frontmatter 的 metadata 字段是一个 JSON 字符串：
            metadata: '{"nanobot": {"requires": {"bins": ["curl"]}, "always": true}}'

        本方法解析这个 JSON，提取 "nanobot" 键下的内容。

        Args:
            raw: metadata 字段的原始 JSON 字符串

        Returns:
            nanobot 元数据字典，解析失败则返回空字典
        """
        try:
            data = json.loads(raw)
            return data.get("nanobot", {}) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _check_requirements(self, skill_meta: dict) -> bool:
        """
        检查技能的依赖是否满足

        依赖类型：
        - bins: 命令行工具（如 curl、gh），通过 shutil.which 检查是否在 PATH 中
        - env: 环境变量（如 API_KEY），通过 os.environ.get 检查是否已设置

        所有依赖都满足才返回 True。

        Args:
            skill_meta: 技能的 nanobot 元数据

        Returns:
            True 表示所有依赖都满足，False 表示有缺失
        """
        requires = skill_meta.get("requires", {})
        # 检查所有必需的命令行工具
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        # 检查所有必需的环境变量
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True
    
    def _get_skill_meta(self, name: str) -> dict:
        """
        获取技能的 nanobot 元数据（从 frontmatter 的 metadata 字段解析）

        Args:
            name: 技能名称

        Returns:
            nanobot 元数据字典
        """
        meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(meta.get("metadata", ""))
    
    def get_always_skills(self) -> list[str]:
        """
        获取所有标记为"始终加载"的技能

        "始终加载"的技能会在每次对话时都注入到 Agent 的上下文中。
        只返回满足依赖要求的技能。

        判断条件（满足任一即可）：
        - frontmatter 中 always: true
        - nanobot metadata 中 always: true

        Returns:
            "始终加载"的技能名称列表
        """
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_nanobot_metadata(meta.get("metadata", ""))
            # 两个地方都可以设置 always 标记
            if skill_meta.get("always") or meta.get("always"):
                result.append(s["name"])
        return result
    
    def get_skill_metadata(self, name: str) -> dict | None:
        """
        从技能文件中解析 YAML frontmatter 元数据

        解析格式：
            ---
            description: "查询天气信息"
            always: true
            metadata: '{"nanobot": {"requires": {"bins": ["curl"]}}}'
            ---

        使用简单的逐行解析（非完整 YAML 解析器），
        以 "key: value" 格式读取每一行。

        Args:
            name: 技能名称

        Returns:
            元数据字典（如 {"description": "查询天气", "always": "true"}），
            文件不存在或无 frontmatter 则返回 None
        """
        content = self.load_skill(name)
        if not content:
            return None
        
        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # 简单的逐行 YAML 解析
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata
        
        return None
