'use strict';

export class PBKDF2 {

    /*
    **
    **
    */
    static async computePBKDF2HexHash(password: string, saltHex: string, iterations: number, hashByteLength: number, hashAlgorithm: string) : Promise<string> {

        const saltBuffer = PBKDF2.hexStringToArrayBuffer(saltHex);
    
        const passwordBuffer = new TextEncoder().encode(password);
    
        const importedKey = await window.crypto.subtle.importKey('raw', passwordBuffer, 'PBKDF2', false, ['deriveBits']);
    
        const hashBuffer = await window.crypto.subtle.deriveBits({
            name: 'PBKDF2',
            salt: saltBuffer,
            iterations: iterations,
            hash: hashAlgorithm,
        }, importedKey, hashByteLength*8);
    
        return PBKDF2.arrayBufferToHexString(hashBuffer);
    }

    /*
    **
    **
    */
    static hexStringToArrayBuffer(input: string) : ArrayBuffer {

        input = input.startsWith('0x') ? input.slice(2) : input;
        
        if (input.length % 2 !== 0)
            input = '0' + input;
        
        const typedArray = new Uint8Array(input.length / 2);
        
        for (let i = 0; i < input.length; i += 2)
            typedArray[i / 2] = parseInt(input.slice(i, i+2), 16);
        
        return typedArray.buffer;
    }

    /*
    **
    **
    */
    static arrayBufferToHexString(input: ArrayBuffer) : string {

        const uint8Array = new Uint8Array(input);
        let hexString = "";
      
        for (let i = 0; i < uint8Array.length; i++) {
            const hexByte = uint8Array[i].toString(16).padStart(2, "0");
            hexString += hexByte;
        }
      
        return hexString;
    }
}