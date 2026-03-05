'use strict';

import { sha256 } from 'js-sha256';
import { customAlphabet } from 'nanoid';

export class Tools {

    /*
    **
    **
    */
    static getDecimal(number: number) : number {

        return parseInt((number % 1).toFixed(2).substring(2));
    }

    /*
    **
    **
    */
    static timestamp() : number {

        return Math.floor(Date.now() / 1000);
    }

    /*
    **
    **
    */
    static sha256(data: any) : string {

        return sha256(data);
    }

    /*
    **
    **
    */
    static isSha256(str: string) : boolean {

        return /^[0-9a-fA-F]{64}$/.test(str);
    }

    /*
    **
    **
    */
    static sleep(time: number) : Promise<void>{

        return new Promise(function (resolve) {
            setTimeout(resolve, time);
        });
    }
    
    /*
    **
    **
    */
    static removeFrom(array: any, value: any) : any {

        for (let i in array) {
            if (array[i] === value) {
                array.splice(i, 1);
                break;
            }
        }
    }

    /*
    **
    **
    */
    static uid(size = 32) : string {

        let alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz-";

        return customAlphabet(alphabet, size)();
    }

    /*
    **
    **
    */
    static randBetween(min: number, max: number) : number {

        return Math.floor(Math.random()*(max-min+1)+min);
    }

    /*
    **
    **
    */
    static isMobile() : boolean {

        if ('ontouchstart' in window)
            return true;

        var prefixes = ' -webkit- -moz- -o- -ms- '.split(' ');
        var query = ['(', prefixes.join('touch-enabled),('), 'heartz', ')'].join('');
        
        return window.matchMedia(query).matches;
    }

    /*
    **
    **
    */
    static getOffset(el) : any {

        let x = 0;
        let y = 0;

        while(el && !isNaN(el.offsetLeft) && !isNaN(el.offsetTop)) {
            x += el.offsetLeft - el.scrollLeft;
            y += el.offsetTop - el.scrollTop;
            el = el.offsetParent;
        }

        return { 
            top: y, 
            left: x
        };
    }

    /*
    **
    **
    */
    static scrollTop() : void {

        if (window.scrollTo)
            window.scrollTo(0, 0);
        else
            document.documentElement.scrollTop = 0;
    }

    /*
    **
    **
    */
    static scrollBottom() : void {

        let scrollingElement = (document.scrollingElement || document.body);
        scrollingElement.scrollTop = scrollingElement.scrollHeight;
    }

    /*
    **
    **
    */
    static humanPeriod(milliseconds: number) : string {

        const units = [
            { label: 'y', ms: 1000 * 60 * 60 * 24 * 365 },
            { label: 'mo', ms: 1000 * 60 * 60 * 24 * 30 },
            { label: 'd', ms: 1000 * 60 * 60 * 24 },
            { label: 'h', ms: 1000 * 60 * 60 },
            { label: 'm', ms: 1000 * 60 }
        ];
    
        const result: string[] = [];
    
        for (let i = 0; i < units.length; i++) {
            if (milliseconds >= units[i].ms) {
                const timeValue = Math.floor(milliseconds / units[i].ms);
                result.push(`${timeValue}${units[i].label}`);
    
                milliseconds -= timeValue * units[i].ms;
            }
    
            if (result.length === 2) break;
        }
    
        if (result.length === 0)
            return "< 1m";
    
        return result.join(' ');
    }

    /*
    **
    **
    */
    static getBase64FromFile(file: File) : Promise<string> {

        return new Promise(function(resolve, reject) {

            let reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = function(event) {
                
                if (reader.result)
                    resolve(reader.result.toString().split('base64,')[1]);
                else
                    reject();
            };
        });
    }

    /*
    **
    **
    */
    static toInputDateFormat(date: Date) : string {

        return date.toISOString().split('T')[0];
    }

    /*
    **
    **
    */
    static base64ToBlob(b64Data, contentType='', sliceSize=512) {

        let byteCharacters = atob(b64Data);
        let byteArrays: any[] = [];
      
        for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {

            let slice = byteCharacters.slice(offset, offset + sliceSize);
        
            let byteNumbers = new Array(slice.length);

            for (let i = 0; i < slice.length; i++)
                byteNumbers[i] = slice.charCodeAt(i);
        
            let byteArray = new Uint8Array(byteNumbers);

            byteArrays.push(byteArray);
        }
      
        let blob = new Blob(byteArrays, {type: contentType});

        return blob;
    }

    /*
    **
    **
    */
    static arrayBufferToString(input: ArrayBuffer) : string {

        return String.fromCharCode.apply(null, new Uint16Array(input));
    }

    /*
    **
    **
    */
    static lightenColor(hex: string, percent: number) : string {

        if (hex.charAt(0) === '#')
            hex = hex.slice(1);
    
        if (hex.length === 3)
            hex = hex.split('').map(c => c + c).join('');
    
        let r = parseInt(hex.substring(0, 2), 16);
        let g = parseInt(hex.substring(2, 4), 16);
        let b = parseInt(hex.substring(4, 6), 16);
    
        r = Math.min(255, Math.floor(r + (255 - r) * (percent / 100)));
        g = Math.min(255, Math.floor(g + (255 - g) * (percent / 100)));
        b = Math.min(255, Math.floor(b + (255 - b) * (percent / 100)));
    
        const newHex = '#' + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('');
    
        return newHex;
    }

    /*
    **
    **
    */
    static downloadBase64(
        filename: string,
        base64: string,
        mime: string = 'application/octet-stream'
    ): void {
        
        const base64Data = base64.includes('base64,')
          ? base64.split('base64,')[1]
          : base64;
      
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);

        for (let i = 0; i < byteCharacters.length; i++)
            byteNumbers[i] = byteCharacters.charCodeAt(i);

        const byteArray = new Uint8Array(byteNumbers);
      
        const blob = new Blob([byteArray], { type: mime });
      
        if (window.navigator && (window.navigator as any).msSaveOrOpenBlob)
            (window.navigator as any).msSaveOrOpenBlob(blob, filename);
        else {
          
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
        
            document.body.appendChild(link);
            link.click();
        
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        }
    }
}