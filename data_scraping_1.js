import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';

for(let i = 1; i < 10; i++) {
  let GPUrl = await getGPFileUrl('https://www.songsterr.com/api/meta/' + i.toString());
  if(GPUrl) {
    console.log(GPUrl)
    fetchDataAndSave(GPUrl, i.toString() + '.gp3')
  }
  else {
    console.log('false')
  }
}

/*
fetchDataAndSave(apiUrl, fileName).then(() => {
    // Do something after saving the file if needed
})
.catch((error) => {
  console.error('Error:', error);
});
*/







async function getGPFileUrl(url) {
  try {
    const response = await fetch(url);

    if (!response.ok) {
      return null;
    }

    const buffer = await response.arrayBuffer();
    const text = new TextDecoder('utf-8').decode(buffer);
    const jsonData = JSON.parse(text);
    //console.log(jsonData)
    if(jsonData.source) {
      //console.log(jsonData.source)
      return jsonData.source;
    }
    else {
      return null;
    }
  } 
  catch (error) {
    return null;
  }
}



async function fetchDataAndSave(url, fileName) {
  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const buffer = await response.buffer();

    let currentDir = path.dirname(new URL(import.meta.url).pathname.replace(/^file:\/\/\//, ''));
    let filePath = path.join(currentDir, '\\tabs\\');
    filePath = path.join(filePath, fileName).replace('E:\\', '');
    fs.writeFileSync(filePath, buffer);
    
    console.log(`File saved successfully at: ${filePath}`);
  } 
  catch (error) {
    console.error('Error fetching and saving file:', error);
    throw error;
  }
}