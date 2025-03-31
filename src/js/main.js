// main.js


function showLoader() {
    document.body.insertAdjacentHTML('afterbegin', '<div class="loader"></div>');
}


function hideLoader() {
    const loader = document.querySelector('.loader');
    if (loader) {
        loader.remove();
    }
}


window.addEventListener('load', () => {
    hideLoader();
});

// Optional: Animate the appearance of cards and forms on page load
document.querySelectorAll('.card, form').forEach((element) => {
    element.style.opacity = '0';
    element.style.transform = 'translateY(20px)';
    setTimeout(() => {
        element.style.transition = 'opacity 0.5s, transform 0.5s';
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
    }, 500);
});
