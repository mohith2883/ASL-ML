function changeModel() {
    let model = document.getElementById("model").value;
    fetch("/set_model", {
        method: "POST",
        body: new URLSearchParams({ model: model }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
    });
}
