window.addEventListener('DOMContentLoaded', async function() {
    
    let data = {
        resources: JSON.parse(document.getElementById('deferred-resources').innerText),
        promises: []
    };

    let head = document.getElementsByTagName('head')[0];
    
    let registerCSSPromise = function(path) {

        data.promises.push(new Promise(async function(resolve) {

            let element = document.createElement('link');
            element.setAttribute('rel', 'stylesheet');
            head.appendChild(element);
            
            element.onload = resolve;
            
            element.href = path;

        }));
    }

    let registerJSPromise = function(path) {

        data.promises.push(new Promise(async function(resolve) {

            let element = document.createElement('script');
            head.appendChild(element);
            
            element.onload = resolve;
            
            element.src = path;

        }));
    }

    for (let path of data.resources.css)
        registerCSSPromise(path);

    if (data.resources.js.length === 1) {

        for (let path of data.resources.js)
            registerJSPromise(path);
        
        await Promise.all(data.promises);
    }
    else {

        for (let path of data.resources.js) {
            registerJSPromise(path);
            await Promise.all(data.promises);
        }
    }

    document.dispatchEvent(new Event('deferredResourcesLoaded'));
});