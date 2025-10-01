import json
import logging
import time


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        return json.dumps(log_entry, ensure_ascii=False)


logging.basicConfig(level=logging.INFO)

root_logger = logging.getLogger()
root_logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setFormatter(JSONFormatter())

root_logger.addHandler(console_handler)

logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
