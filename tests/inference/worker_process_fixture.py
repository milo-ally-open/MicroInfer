"""供 multiprocessing worker contract 測試載入的純 Python Worker。"""


class EchoWorker:
    def start(self) -> None:
        self.started = True

    def invoke(self, operation: str, *args, **kwargs):
        return {"operation": operation, "args": args, "kwargs": kwargs}

    def health_info(self):
        return {"backend": "process-test"}

    def gpu_status(self):
        return {"gpu": "child"}

    def close(self) -> None:
        self.started = False


def create_worker() -> EchoWorker:
    return EchoWorker()
