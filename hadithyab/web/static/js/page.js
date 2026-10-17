// حدیث‌یاب reading pages (Quran, Nahj al-Balagha, hadith lists, about): theme and donate only.
"use strict";

document.getElementById("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* private mode */ }
});

document.getElementById("copy-card").addEventListener("click", async (e) => {
  try { await navigator.clipboard.writeText("6219861967576876"); e.target.textContent = "کپی شد"; } catch (err) { /* no clipboard */ }
  setTimeout(() => { e.target.textContent = "کپی شماره کارت"; }, 2000);
});
