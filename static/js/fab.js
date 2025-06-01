document.addEventListener('DOMContentLoaded', function() {
  const fabButton = document.getElementById('fab');
  if (fabButton) {
    fabButton.addEventListener('click', function(event) {
      event.preventDefault(); // Prevent default anchor behavior
      // Construct the URL for the support page.
      // Assuming 'support.html' will be served at a '/support' route.
      // This might need adjustment based on the actual Flask routing.
      window.location.href = '/support';
    });
  } else {
    console.error('FAB element not found.');
  }
});
