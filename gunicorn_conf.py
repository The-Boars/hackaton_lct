import multiprocessing
# Под CPU-bound инференс чаще лучше несколько воркеров (по числу ядер или ядер/2).
workers = int(multiprocessing.cpu_count() * 0.75) or 1
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8080"
# Чуть увеличенный keepalive улучшает повторные соединения
keepalive = 65
# Ограничим размер запроса (DoS-защита)
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190
# Логи — по вкусу
accesslog = "-"
errorlog = "-"
