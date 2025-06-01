document.addEventListener('DOMContentLoaded', function() {
    const copyButton = document.getElementById('copyCardNumberBtn');
    const feedbackSpan = document.getElementById('copyFeedback');
    const cardNumberElement = document.getElementById('cardNumber');

    if (copyButton && cardNumberElement && feedbackSpan) {
        copyButton.addEventListener('click', function() {
            const cardNumber = cardNumberElement.innerText.replace(/-/g, ''); // Remove hyphens for copying if desired, or keep them.

            navigator.clipboard.writeText(cardNumber).then(function() {
                feedbackSpan.textContent = 'شماره کارت کپی شد!';
                feedbackSpan.style.display = 'inline'; // Make it visible

                setTimeout(function() {
                    feedbackSpan.textContent = '';
                    feedbackSpan.style.display = 'none'; // Hide it again
                }, 3000); // Hide after 3 seconds
            }).catch(function(error) {
                console.error('خطا در کپی کردن شماره کارت: ', error);
                feedbackSpan.textContent = 'خطا در کپی!';
                feedbackSpan.style.display = 'inline';
                feedbackSpan.style.color = 'red';


                setTimeout(function() {
                    feedbackSpan.textContent = '';
                    feedbackSpan.style.display = 'none';
                    feedbackSpan.style.color = ''; // Reset color
                }, 3000);
            });
        });
    } else {
        if (!copyButton) console.error('Button with id "copyCardNumberBtn" not found.');
        if (!cardNumberElement) console.error('Element with id "cardNumber" not found.');
        if (!feedbackSpan) console.error('Element with id "copyFeedback" not found.');
    }
});
