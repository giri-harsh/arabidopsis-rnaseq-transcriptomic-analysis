/* ==========================================================================
   Signature element: procedurally-plotted "volcano scatter" in the hero,
   built from a small seeded dataset shaped like real DESeq2 output
   (log2FoldChange vs -log10(padj)). Points animate in on load, echoing the
   pipeline's actual primary output figure rather than a generic hero.
   ========================================================================== */

(function () {
  const svg = document.getElementById("volcano-svg");
  if (!svg) return;

  const W = 480, H = 320, PAD = 36;
  const xMax = 4.2, yMax = 11;

  function seededRandom(seed) {
    let s = seed;
    return function () {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  }
  const rnd = seededRandom(42);

  function scaleX(v) { return PAD + ((v + xMax) / (2 * xMax)) * (W - 2 * PAD); }
  function scaleY(v) { return H - PAD - (v / yMax) * (H - 2 * PAD); }

  const points = [];
  const nNS = 90, nUp = 22, nDown = 22;

  for (let i = 0; i < nNS; i++) {
    const lfc = (rnd() - 0.5) * 1.6;
    const y = rnd() * 1.3;
    points.push({ lfc, y, status: "ns" });
  }
  for (let i = 0; i < nUp; i++) {
    const lfc = 1.1 + rnd() * 2.6;
    const y = 1.3 + rnd() * 8.5;
    points.push({ lfc, y, status: "up" });
  }
  for (let i = 0; i < nDown; i++) {
    const lfc = -1.1 - rnd() * 2.6;
    const y = 1.3 + rnd() * 8.5;
    points.push({ lfc, y, status: "down" });
  }

  const colors = { ns: "#4d5a58", up: "#c06a4d", down: "#6fae87" };

  const ns = "http://www.w3.org/2000/svg";
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // threshold guide lines
  [1.3, xMax].forEach(() => {});
  const hLine = document.createElementNS(ns, "line");
  hLine.setAttribute("x1", PAD); hLine.setAttribute("x2", W - PAD);
  hLine.setAttribute("y1", scaleY(1.3)); hLine.setAttribute("y2", scaleY(1.3));
  hLine.setAttribute("stroke", "#2a3438"); hLine.setAttribute("stroke-dasharray", "4,4");
  svg.appendChild(hLine);

  [1.1, -1.1].forEach((v) => {
    const vLine = document.createElementNS(ns, "line");
    vLine.setAttribute("x1", scaleX(v)); vLine.setAttribute("x2", scaleX(v));
    vLine.setAttribute("y1", PAD); vLine.setAttribute("y2", H - PAD);
    vLine.setAttribute("stroke", "#2a3438"); vLine.setAttribute("stroke-dasharray", "4,4");
    svg.appendChild(vLine);
  });

  points.forEach((p, i) => {
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", scaleX(p.lfc).toFixed(1));
    c.setAttribute("cy", scaleY(p.y).toFixed(1));
    c.setAttribute("r", p.status === "ns" ? 2.4 : 3.4);
    c.setAttribute("fill", colors[p.status]);
    c.classList.add("v-point");
    c.style.animationDelay = `${(i * 9) % 700}ms`;
    svg.appendChild(c);
    requestAnimationFrame(() => c.classList.add("show"));
  });

  const axisLabelX = document.createElementNS(ns, "text");
  axisLabelX.setAttribute("x", W / 2);
  axisLabelX.setAttribute("y", H - 6);
  axisLabelX.setAttribute("fill", "#7d908c");
  axisLabelX.setAttribute("font-size", "10");
  axisLabelX.setAttribute("font-family", "SFMono-Regular, Menlo, monospace");
  axisLabelX.setAttribute("text-anchor", "middle");
  axisLabelX.textContent = "log2 fold change";
  svg.appendChild(axisLabelX);

  const axisLabelY = document.createElementNS(ns, "text");
  axisLabelY.setAttribute("x", -H / 2);
  axisLabelY.setAttribute("y", 12);
  axisLabelY.setAttribute("fill", "#7d908c");
  axisLabelY.setAttribute("font-size", "10");
  axisLabelY.setAttribute("font-family", "SFMono-Regular, Menlo, monospace");
  axisLabelY.setAttribute("text-anchor", "middle");
  axisLabelY.setAttribute("transform", "rotate(-90)");
  axisLabelY.textContent = "-log10(padj)";
  svg.appendChild(axisLabelY);
})();

/* Reveal sections gently on scroll */
(function () {
  const targets = document.querySelectorAll(".obj-card, .gallery-card, .pstep");
  if (!("IntersectionObserver" in window) || targets.length === 0) return;

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.transition = "opacity 0.5s ease, transform 0.5s ease";
          entry.target.style.opacity = "1";
          entry.target.style.transform = "translateY(0)";
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  targets.forEach((el) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(12px)";
    io.observe(el);
  });
})();
