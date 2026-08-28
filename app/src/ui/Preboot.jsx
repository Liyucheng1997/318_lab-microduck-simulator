// Pre-boot veil: covers everything until the title page's logo and fonts
// are in, so the menu never pops in half-assembled. Light grey spinner,
// centered on the ink.
import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import { keyframes } from "@mui/material/styles";
import { useGame } from "../store.js";
import { INK } from "../theme.js";

const spin = keyframes`to { transform: rotate(360deg); }`;

export default function Preboot() {
  const prebootDone = useGame((s) => s.prebootDone);
  const [gone, setGone] = useState(false);

  useEffect(() => {
    if (!prebootDone) return;
    const t = setTimeout(() => setGone(true), 300);
    return () => clearTimeout(t);
  }, [prebootDone]);

  if (gone) return null;
  return (
    <Box
      aria-hidden
      sx={{
        position: "fixed",
        inset: 0,
        zIndex: 40,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: INK,
        opacity: prebootDone ? 0 : 1,
        pointerEvents: prebootDone ? "none" : "auto",
        transition: "opacity 0.25s ease",
      }}
    >
      <Box
        sx={{
          width: 22,
          height: 22,
          borderRadius: "50%",
          border: "2px solid rgba(255, 255, 255, 0.14)",
          borderTopColor: "rgba(255, 255, 255, 0.55)",
          animation: `${spin} 0.8s linear infinite`,
          "@media (prefers-reduced-motion: reduce)": {
            animationDuration: "1.6s",
          },
        }}
      />
    </Box>
  );
}
