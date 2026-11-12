// The theme lays out the page after Plotly sizes its figures, which then
// overflow into the sidebar until the window is resized, so resize it once
document.addEventListener("DOMContentLoaded", () => {
  window.dispatchEvent(new Event("resize"));
});
