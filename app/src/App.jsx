// App shell: layers in z order - scene canvas (1), halftone (2), CRT (5),
// HUD + touch overlay (9-12), BIOS readout (20), title menu (30), preboot
// veil (40). Also owns the pre-game device detection (gamepad, touch) and
// the preboot -> title -> enter flow.
import { useEffect } from "react";
import GameCanvas from "./scene/GameCanvas.jsx";
import { Halftone, CrtOverlay } from "./ui/Overlays.jsx";
import Hud from "./ui/Hud.jsx";
import TouchOverlay from "./ui/TouchOverlay.jsx";
import BiosOverlay from "./ui/BiosOverlay.jsx";
import TitleMenu from "./ui/TitleMenu.jsx";
import Preboot from "./ui/Preboot.jsx";
import { useGame } from "./store.js";
import { signed } from "./game/signed.js";

export default function App() {
  useEffect(() => {
    const params = new URLSearchParams(location.search);

    // Gamepad presence for the tutorial variant, before the game controller
    // takes over polling (the game loop keeps mirroring it afterwards).
    const pollPad = () => {
      const gp = [...(navigator.getGamepads?.() ?? [])].find((p) => p && p.connected);
      useGame.setState({ padConnected: !!gp });
    };
    window.addEventListener("gamepadconnected", pollPad);
    window.addEventListener("gamepaddisconnected", pollPad);
    pollPad();

    // Touch detection, latching: (pointer: coarse) OR any real touch proves
    // the device (DevTools device modes and hybrid laptops often fail the
    // media query). __microduckTouched hands the proof to controls/touch.js
    // which boots later. ?touch=1 forces the thumbs on any device.
    const coarse = matchMedia("(pointer: coarse)");
    const armTouch = () => {
      window.__microduckTouched = true;
      useGame.setState({ touchMode: true });
    };
    const onCoarse = () => coarse.matches && armTouch();
    if (coarse.matches) armTouch();
    else coarse.addEventListener("change", onCoarse);
    window.addEventListener("touchstart", armTouch, { once: true, passive: true });
    if (params.get("touch") === "1") armTouch();

    // Preboot veil: hold the title page until its logo and the fonts are
    // in (grey spinner meanwhile), capped so a broken asset can never wall
    // the menu off. ?boot=1 skips the title entirely - testing hook.
    if (params.get("boot") === "1") {
      useGame.setState({ prebootDone: true, entered: true, menuOpen: false });
    } else {
      const img = new Image();
      img.src = signed("./assets/duck-head-mark.webp");
      const imgReady = new Promise((res) => {
        if (img.complete && img.naturalWidth) res();
        else {
          img.addEventListener("load", res, { once: true });
          img.addEventListener("error", res, { once: true });
        }
      });
      const fontsReady = document.fonts?.ready ?? Promise.resolve();
      const cap = new Promise((res) => setTimeout(res, 2500));
      Promise.race([Promise.all([imgReady, fontsReady]), cap]).then(() => {
        useGame.setState({ prebootDone: true, menuOpen: true });
      });
    }

    return () => {
      window.removeEventListener("gamepadconnected", pollPad);
      window.removeEventListener("gamepaddisconnected", pollPad);
      coarse.removeEventListener("change", onCoarse);
      window.removeEventListener("touchstart", armTouch);
    };
  }, []);

  return (
    <>
      <GameCanvas />
      <Halftone />
      <CrtOverlay />
      <Hud />
      <TouchOverlay />
      <BiosOverlay />
      <TitleMenu />
      <Preboot />
    </>
  );
}
