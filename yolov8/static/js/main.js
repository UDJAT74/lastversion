let imgs=document.querySelectorAll("img");
imgs.forEach(img =>{
    img.addEventListener("click",()=>{
        img.style.width="100%";
        img.style.height="650px";
    })
})