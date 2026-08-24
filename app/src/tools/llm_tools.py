


def extract_text_content(content) -> str:
        if not content:
            return ""

        # OpenAI / string content
        if isinstance(content, str):
            return content

        # Gemini / multimodal content
        if isinstance(content, list):
            texts = []

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "text":
                    text = item.get("text")

                    if text:
                        texts.append(text)

            return "\n".join(texts)

        return str(content) 