document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const resultList = document.getElementById('result-list');
    const textQueryInput = document.getElementById('text-query');
    const imageQueryInput = document.getElementById('image-query');
    const lambdaInput = document.getElementById('lambda');
    const lambdaDisplay = document.getElementById('lambda-display');
    const embeddingChoiceInput = document.getElementById('embedding-choice');
    const pcaKGroup = document.getElementById('pca-k-group');

    // Display lambda value on input change
    lambdaInput.addEventListener('input', () => {
        lambdaDisplay.textContent = `Text Influence (If Hybrid): ${lambdaInput.value}`;
    });

    // Initial lambda display update
    lambdaDisplay.textContent = `Text Influence (If Hybrid): ${lambdaInput.value}`;

    // Show PCA component input when PCA embedding is selected
    embeddingChoiceInput.addEventListener('change', () => {
        pcaKGroup.style.display = embeddingChoiceInput.value === 'pca' ? 'block' : 'none';
    });

    // Initial check for PCA visibility
    if (embeddingChoiceInput.value === 'pca') {
        pcaKGroup.style.display = 'block';
    }

    // Handle form submission
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        resultList.innerHTML = 'Searching...';

        const formData = new FormData();
        
        // Append text query and image query data
        if (textQueryInput.value) formData.append('text_query', textQueryInput.value);
        if (imageQueryInput.files.length > 0) formData.append('image_query', imageQueryInput.files[0]);
        formData.append('lambda', lambdaInput.value);
        formData.append('embedding_choice', embeddingChoiceInput.value);
        formData.append('pca_k', document.getElementById('pca-k').value);

        // Fetch search results
        fetch('/search', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                resultList.innerHTML = '';
                if (data.results && data.results.length > 0) {
                    const resultGrid = document.createElement('div');
                    resultGrid.className = 'result-grid';

                    data.results.forEach((result, index) => {
                        const resultItem = document.createElement('div');
                        resultItem.className = 'result-item';

                        const img = document.createElement('img');
                        img.src = result.image_path;
                        img.alt = `Result ${index + 1}`;

                        const similarityText = document.createElement('p');
                        similarityText.textContent = `Similarity: ${(result.similarity * 100).toFixed(2)}%`;

                        resultItem.appendChild(img);
                        resultItem.appendChild(similarityText);
                        resultGrid.appendChild(resultItem);
                    });

                    resultList.appendChild(resultGrid);
                } else {
                    resultList.textContent = 'No results found.';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                resultList.textContent = 'An error occurred during search.';
            });
    });
});
