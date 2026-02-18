# Use an official Python image with uv pre-installed (adjust 3.12 to your Python version)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set the working directory inside the container
WORKDIR /app

# Enable bytecode compilation for slightly faster application startup
ENV UV_COMPILE_BYTECODE=1

# Tell uv to copy files instead of using hardlinks (which don't work well across Docker mounts)
ENV UV_LINK_MODE=copy

# 1. Copy only the dependency files first
COPY pyproject.toml uv.lock ./

# 2. Install dependencies (without the project code or dev dependencies)
# The cache mount speeds up subsequent builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 3. Copy the rest of your application code
COPY . .

# 4. Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Put the virtual environment on the path so `python` automatically uses it
ENV PATH="/app/.venv/bin:$PATH"

# Replace `main.py` with whatever script starts your application
CMD ["python", "src/run.py"]