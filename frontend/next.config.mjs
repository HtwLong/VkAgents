/** @type {import('next').NextConfig} */
const nextConfig = {
  // Normal framework output is the safe default for Vercel. Only the Docker
  // build explicitly opts into Next's generated standalone server.
  output: process.env.BUILD_STANDALONE === "true" ? "standalone" : undefined,

  experimental: {
    proxyTimeout: 16000_000,
  },

  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    const apiBase =
      process.env.BACKEND_INTERNAL_URL ?? "http://127.0.0.1:8000"

    return [
      {
        source: "/api/backend/:path*",
        destination: `${apiBase}/:path*`,
      },
    ]
  },
}

export default nextConfig
