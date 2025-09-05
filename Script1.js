
document.addEventListener('DOMContentLoaded', function() {
    var fileInput = document.getElementById('fileInput');
    var fileNameDisplay = document.getElementById('imageview');

    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            var fileName = fileInput.files[0].name;
            fileNameDisplay.textContent = fileName + " uploaded successfully";
            fileNameDisplay.style.fontSize = "15px";
            fileNameDisplay.style.paddingTop = "20px";
            
        } else {
            fileNameDisplay.textContent = 'No file selected';
        }
    });
});

document.getElementById("uploadButton").addEventListener("click", () => {
    const fileInput = document.getElementById("fileInput");
    const urlInput = document.querySelector(".url-input").value;
    if (fileInput.files.length !== 0 && urlInput !== "") {
        // alert("Please do one thing. Either select an image or enter a URL");
        return false;
    } else if (urlInput !== "") {
        // alert(`File URL added successfully.`);
    } else if (fileInput.files.length !== 0) {
        // alert(`File selected successfully.`);
    } else {
        // alert("Please choose a file to upload or provide a file URL.")
        return false;
    }
});



