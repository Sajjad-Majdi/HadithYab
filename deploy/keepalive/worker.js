// Keeps the Render free instance awake: it sleeps after 15 idle minutes and
// then takes ~50 s to wake and reload the index. A ping every 13 minutes
// avoids that; */13 fires at :00 :13 :26 :39 :52, so no gap reaches 15. One always-on service uses at most 744 of Render's 750 free
// hours a month.
const TARGET = "https://hadithyab.onrender.com/health";

export default {
  async scheduled(_event, _env, ctx) {
    ctx.waitUntil(fetch(TARGET, { headers: { "user-agent": "hadithyab-keepalive" } }));
  },
  async fetch() {
    return new Response("hadithyab keepalive", { headers: { "content-type": "text/plain" } });
  },
};
