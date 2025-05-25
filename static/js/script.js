document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Real-time BMI validation
    const bmiInput = document.getElementById('bmi');
    if (bmiInput) {
        bmiInput.addEventListener('input', function() {
            const bmi = parseFloat(this.value);
            if (bmi < 10 || bmi > 60) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    }

    // Real-time glucose level validation
    const glucoseInput = document.getElementById('avg_glucose_level');
    if (glucoseInput) {
        glucoseInput.addEventListener('input', function() {
            const glucose = parseFloat(this.value);
            if (glucose < 20 || glucose > 500) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    }
});