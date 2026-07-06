/*
 * ConstructIQ — shared frontend helpers (vanilla JS, no build step).
 * Include on every page BEFORE the page's inline script:
 *   <script src="js/api.js"></script>
 *
 * Everything is attached to window. Interface:
 *
 *   API                     -> "" (same-origin base for all requests)
 *   getProjectId()          -> string  (localStorage-persisted, default "PRJ-2024-001")
 *   setProjectId(id)        -> void
 *   apiGet(path)            -> Promise<json>  (throws Error(detail) on non-2xx)
 *   apiPost(path, body)     -> Promise<json>  (JSON headers; throws Error(detail) on non-2xx)
 *   apiUpload(path, formData) -> Promise<json> (multipart; throws Error(detail) on non-2xx)
 *   fmtINR(n)               -> "₹4,12,000" style string (en-IN grouping, no decimals
 *                              for whole numbers, up to 2 decimals otherwise; "—" for
 *                              null/undefined/non-numeric)
 *   fmtDate(iso)            -> "Aug 12, 2026" style string ("—" for empty input)
 *   toast(message, kind)    -> floating self-removing card, bottom-right.
 *                              kind: "info" (charcoal #1F2937) | "success" (#16a34a)
 *                                    | "error" (#ba1a1a). Defaults to "info".
 *   wireNav(active)         -> highlights the top-nav link whose data-nav attribute
 *                              equals `active` (safety-orange #F97316 text + underline).
 *                              Valid keys: "chat","risk","vendors","negotiation","po".
 *   escapeHtml(str)         -> HTML-escapes &, <, >, ", ' — use for ANY user/API text
 *                              interpolated into an innerHTML template string.
 *   starsHtml(rating)       -> plain-text 5-star string, e.g. starsHtml(4) === "★★★★☆"
 *                              (rating rounded to nearest whole star, clamped 0–5).
 */
(function () {
  "use strict";

  // Same-origin base for all API calls.
  const API = "";

  const PROJECT_KEY = "constructiq_project_id";
  const DEFAULT_PROJECT = "PRJ-2024-001";

  // ---------------------------------------------------------------- project id

  function getProjectId() {
    try {
      return localStorage.getItem(PROJECT_KEY) || DEFAULT_PROJECT;
    } catch (e) {
      return DEFAULT_PROJECT;
    }
  }

  function setProjectId(id) {
    try {
      localStorage.setItem(PROJECT_KEY, String(id));
    } catch (e) {
      /* storage unavailable — ignore, getProjectId falls back to default */
    }
  }

  // ---------------------------------------------------------------- http core

  // Shared response handling: 2xx -> parsed JSON; non-2xx -> throw Error whose
  // message is the FastAPI {"detail": ...} field when the body is JSON, else
  // the HTTP status text.
  function request(path, options) {
    return fetch(API + path, options)
      .then(function (res) {
        if (res.ok) {
          return res.json().catch(function () {
            return {};
          });
        }
        return res
          .json()
          .catch(function () {
            return null;
          })
          .then(function (data) {
            let detail;
            if (data && data.detail !== undefined && data.detail !== null) {
              detail =
                typeof data.detail === "string"
                  ? data.detail
                  : JSON.stringify(data.detail);
            } else {
              detail = res.statusText || "Request failed (HTTP " + res.status + ")";
            }
            throw new Error(detail);
          });
      })
      .catch(function (err) {
        // Normalise network-level failures (fetch rejects with TypeError).
        if (err instanceof Error && err.message) throw err;
        throw new Error("Network error — could not reach the ConstructIQ API");
      });
  }

  function apiGet(path) {
    return request(path, {
      method: "GET",
      headers: { Accept: "application/json" },
    });
  }

  function apiPost(path, body) {
    return request(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body === undefined ? {} : body),
    });
  }

  function apiUpload(path, formData) {
    // NOTE: no Content-Type header — the browser sets the multipart boundary.
    return request(path, { method: "POST", body: formData });
  }

  // ---------------------------------------------------------------- formatters

  function fmtINR(n) {
    const num = typeof n === "string" && n.trim() !== "" ? Number(n) : n;
    if (typeof num !== "number" || !isFinite(num)) return "—";
    const isWhole = Math.abs(num - Math.round(num)) < 1e-9;
    return (
      "₹" +
      num.toLocaleString("en-IN", {
        minimumFractionDigits: 0,
        maximumFractionDigits: isWhole ? 0 : 2,
      })
    );
  }

  function fmtDate(iso) {
    if (iso === null || iso === undefined || iso === "") return "—";
    let d;
    // Parse date-only strings ("2026-08-12") as LOCAL dates to avoid the
    // UTC-midnight off-by-one-day problem.
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(iso).trim());
    if (m) {
      d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
    } else {
      d = new Date(iso);
    }
    if (isNaN(d.getTime())) return String(iso);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  // ---------------------------------------------------------------- escaping

  function escapeHtml(str) {
    return String(str === null || str === undefined ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ---------------------------------------------------------------- toast

  const TOAST_COLORS = {
    info: "#1F2937",
    success: "#16a34a",
    error: "#ba1a1a",
  };

  function toastHost() {
    let host = document.getElementById("ciq-toast-host");
    if (!host) {
      host = document.createElement("div");
      host.id = "ciq-toast-host";
      host.style.cssText =
        "position:fixed;bottom:24px;right:24px;z-index:9999;" +
        "display:flex;flex-direction:column;gap:8px;align-items:flex-end;" +
        "pointer-events:none;";
      document.body.appendChild(host);
    }
    return host;
  }

  function toast(message, kind) {
    const color = TOAST_COLORS[kind] || TOAST_COLORS.info;
    const card = document.createElement("div");
    card.setAttribute("role", "status");
    card.style.cssText =
      "background:" + color + ";color:#ffffff;" +
      "font-family:'JetBrains Mono',ui-monospace,monospace;" +
      "padding:12px 16px;border-radius:8px;max-width:380px;" +
      "box-shadow:0 4px 12px rgba(31,41,55,0.25);pointer-events:auto;" +
      "opacity:0;transform:translateY(8px);" +
      "transition:opacity .25s ease,transform .25s ease;";

    const label = document.createElement("div");
    label.textContent = (kind || "info").toUpperCase();
    label.style.cssText =
      "font-size:10px;font-weight:600;letter-spacing:0.1em;" +
      "text-transform:uppercase;opacity:0.75;margin-bottom:4px;";

    const body = document.createElement("div");
    body.textContent = String(message === null || message === undefined ? "" : message);
    body.style.cssText = "font-size:12px;font-weight:500;line-height:18px;word-break:break-word;";

    card.appendChild(label);
    card.appendChild(body);
    toastHost().appendChild(card);

    requestAnimationFrame(function () {
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";
    });

    setTimeout(function () {
      card.style.opacity = "0";
      card.style.transform = "translateY(8px)";
      setTimeout(function () {
        if (card.parentNode) card.parentNode.removeChild(card);
      }, 300);
    }, 4200);
  }

  // ---------------------------------------------------------------- nav

  const ACTIVE_ORANGE = "#F97316";

  function wireNav(active) {
    const links = document.querySelectorAll("[data-nav]");
    for (let i = 0; i < links.length; i++) {
      const a = links[i];
      if (a.getAttribute("data-nav") === active) {
        a.style.color = ACTIVE_ORANGE;
        a.style.borderBottom = "2px solid " + ACTIVE_ORANGE;
        a.style.fontWeight = "700";
        a.setAttribute("aria-current", "page");
      } else {
        a.style.borderBottom = "2px solid transparent";
        a.removeAttribute("aria-current");
      }
    }
  }

  // ---------------------------------------------------------------- stars

  function starsHtml(rating) {
    let r = Math.round(Number(rating));
    if (!isFinite(r)) r = 0;
    if (r < 0) r = 0;
    if (r > 5) r = 5;
    let out = "";
    for (let i = 0; i < 5; i++) out += i < r ? "★" : "☆"; // ★ / ☆
    return out;
  }

  // ---------------------------------------------------------------- exports

  window.API = API;
  window.getProjectId = getProjectId;
  window.setProjectId = setProjectId;
  window.apiGet = apiGet;
  window.apiPost = apiPost;
  window.apiUpload = apiUpload;
  window.fmtINR = fmtINR;
  window.fmtDate = fmtDate;
  window.escapeHtml = escapeHtml;
  window.toast = toast;
  window.wireNav = wireNav;
  window.starsHtml = starsHtml;
})();
