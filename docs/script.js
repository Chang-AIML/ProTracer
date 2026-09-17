const copyButton = document.querySelector('#copy-button');
const bibtex = document.querySelector('#bibtex-code');

copyButton?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent);
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
  } catch {
    copyButton.textContent = 'Select text to copy';
  }
});

const carousel = document.querySelector('[data-carousel]');

if (carousel) {
  const slides = [...carousel.querySelectorAll('[data-slide]')];
  const dots = [...carousel.querySelectorAll('[data-dot]')];
  const count = carousel.querySelector('[data-count]');
  let current = 0;

  const showSlide = (nextIndex) => {
    const previousVideo = slides[current].querySelector('video');
    previousVideo?.pause();
    current = (nextIndex + slides.length) % slides.length;
    slides.forEach((slide, index) => { slide.hidden = index !== current; });
    dots.forEach((dot, index) => dot.classList.toggle('active', index === current));
    count.textContent = `${current + 1} / ${slides.length}`;
    const activeVideo = slides[current].querySelector('video');
    activeVideo.currentTime = 0;
    activeVideo.play().catch(() => {});
  };

  dots.forEach((dot, index) => dot.addEventListener('click', () => showSlide(index)));
  slides[0].querySelector('video')?.play().catch(() => {});
}
