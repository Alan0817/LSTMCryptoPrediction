## Python Coding Style

Follow the existing repository coding style. Optimize for readability and
maintainability, not minimum line count.

Required conventions:

- Use one import per module. Do not write combined imports such as:
  `import os, requests`.
- Use normal multi-line Python formatting.
- Do not place multiple statements or dataclass fields on one line.
- Include spaces around type annotations:
  `title: str`, not `title:str`.
- Do not use semicolons to compress statements.
- Do not write non-trivial function or method bodies on the same line as
  the definition.
- Prefer:

      def to_dict(self):
          return asdict(self)

  over:

      def to_dict(self): return asdict(self)

- Keep blank-line spacing consistent with the existing source files.
- Preserve the style of nearby modules when adding or modifying code.
- Do not perform unrelated style rewrites.
- Before finishing, review all modified Python files for style consistency.
- When unsure about formatting, inspect existing files in src/ and follow
their prevailing style rather than introducing a new compact style.