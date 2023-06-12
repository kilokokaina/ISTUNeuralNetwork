let answerModal = new bootstrap.Modal(document.getElementById('answerModal'));

async function sendRequest() {
    let image = document.getElementById('inputFile').files[0];
    let formData = new FormData();

    formData.append("inputFile", image);

    let response = await fetch('/query', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        let responseText = await response.text();
        document.getElementById('answerPlaceholder').innerText = 'Ответ нейросети: ' + responseText;

        answerModal.show();
    } else {
        console.log('server error');
    }
}