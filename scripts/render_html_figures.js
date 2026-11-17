// Render docs/figures/src/*.html to PNG (cover at 560x280 @2x, architecture at 1100x600 @2x).
//   node scripts/render_html_figures.js
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const jobs = [['cover', 560, 280], ['architecture', 1100, 600]];
  for (const [name, w, h] of jobs) {
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await page.goto('file://' + path.resolve(__dirname, '../docs/figures/src', name + '.html'));
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.resolve(__dirname, '../docs/figures', name + '.png') });
    console.log('wrote', name);
  }
  await browser.close();
})();
