/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          50:  "#F3F6FA",
          100: "#E1E8F2",
          200: "#C2D0E2",
          300: "#8FA3BF",
          400: "#5C78A0",
          500: "#3A5680",
          600: "#1E4275",
          700: "#0F2845",
          800: "#0A1E36",
          900: "#061425",
        },
        gold: {
          50:  "#FBF6EC",
          100: "#F5E8C8",
          200: "#EBCF8E",
          300: "#DCB45A",
          400: "#C9963A",
          500: "#B07E26",
          600: "#8B6320",
          700: "#654917",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        serif: ["Georgia", "Times New Roman", "serif"],
      },
    },
  },
  plugins: [],
};
