# Running the application with Docker Compose

The root `compose.yaml` builds and runs both applications:

- frontend: <http://localhost:3000>
- backend API: <http://localhost:8000> (documentation at `/docs`)

The frontend sends API requests through its own `/api/backend` proxy. Inside the
Compose network, that proxy reaches the backend using the service name `backend`.

## First start

Install Docker Engine, Docker Compose, the NVIDIA driver, and NVIDIA Container
Toolkit on the host. Then, from the repository root:

```sh
cp .env.example .env
```

Put the real `OPENAI_API_KEY` in `.env`. Select a GPU with
`NVIDIA_VISIBLE_DEVICES=all`, a zero-based index such as `0`, or its GPU UUID.

```sh
docker compose up -d --build
docker compose ps
docker compose logs -f
```

Use `docker compose logs -f backend` or `docker compose logs -f frontend` to
follow only one service. Press Ctrl+C to stop following logs; the containers keep
running.

## Normal operation

```sh
# Start containers that already exist
docker compose up -d

# Stop without deleting the containers
docker compose stop

# Stop and remove the containers and network; named volumes are retained
docker compose down

# Rebuild after pulling or changing code
git pull
docker compose up -d --build
```

Do not add `--volumes` to `docker compose down` unless the saved runs and dataset
cache should also be deleted.

## Move to another Unix/Linux computer

The usual approach is to push/clone the Git repository on the destination, copy
`.env` through a secure channel, install the Docker/NVIDIA prerequisites, and run:

```sh
docker compose up -d --build
```

The destination GPU will generally have a different UUID, so update
`NVIDIA_VISIBLE_DEVICES` there. `gpus: all` requires an NVIDIA GPU and a working
NVIDIA Container Toolkit installation.

Git does not transfer Docker volumes. To move existing run artifacts and cached
datasets too, archive the two named volumes separately or copy the relevant data
from a stopped backend container. If only the application is being moved, let
Compose create empty volumes on the new host.

To move prebuilt images instead of rebuilding them, export both images:

```sh
docker image save vkagents-backend:latest vkagents-frontend:latest -o vkagents-images.tar
```

Transfer the tar file and repository to the destination, then run:

```sh
docker image load -i vkagents-images.tar
docker compose up -d --no-build
```

Images are architecture-specific unless built as multi-platform images. Rebuild
on the destination when moving between x86-64 and ARM64.
