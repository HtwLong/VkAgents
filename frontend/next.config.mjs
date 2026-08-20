/** @type {import('next').NextConfig} */
const nextConfig = {
  // The Docker image runs Next's generated standalone server. Vercel uses its
  // own Next.js build adapter and should receive the normal framework output.
  output: process.env.VERCEL === "1" ? undefined : "standalone",

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
