"""推论层错误，不依赖 HTTP 框架。"""


class InferenceError(Exception):
    """可映射至 API 错误回应的推论错误。"""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
