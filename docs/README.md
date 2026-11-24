# Paper project page

Buildless static project page for **On Neural Scaling Laws for Weather Emulation through Continual Training**. No package installation, externally loaded fonts, analytics, or build step is required.

## Inspect locally

From the code repository root (`neural-scaling-weather`):

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8000>. Stop the server with Ctrl+C. Alternatively, open `docs/index.html` directly; the content and figure viewer work without a server. Clipboard availability depends on the browser.

The entire page lives in this folder. `index.html` contains the research text, `styles.css` controls the responsive layout, and `script.js` handles result tabs, figure enlargement, and citation copying. Without JavaScript, both result figures remain visible and figure links open the original image. Figures are stored locally.

## Publication status

Prepared for a future GitHub Pages site served from this repository's `docs/` folder. `.nojekyll` allows the static files to be served directly. **Nothing has been pushed, published, or configured on GitHub.** No deployment workflow is included. Review the page before enabling publishing. The expected project URL, if published under the current owner/repository, is `https://shashanksubramanian.github.io/neural-scaling-weather/` (also used for the social preview image metadata).

Typography uses system fonts; there are no font files or external font requests.
