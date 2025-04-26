build:
	docker-compose up --build
down:
	docker-compose down
exec-mlflow:
	docker exec -it app-mlflow-1 /bin/bash
fastapi:
	fastapi dev api/main.py
mlflow-server:
	cd mlflow/ && mlflow server --host 127.0.0.1 --port 5000
