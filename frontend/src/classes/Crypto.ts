'use strict';

import { Client } from '@src/classes';

export interface CryptoConfigs {
    [key: string]: CryptoConfigEntry
}

export interface CryptoConfigEntry {
    rawKey: string,
    derivedKeySalt: number[],
    derivedKeyByteLength: number,
    derivedKeyIterations: number,
    ivByteLength: number
}

declare const __STORE_Crypto__: number[];

export class Crypto {

    private static BASE64_ENCODINGS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';

    private lib = window.crypto.subtle;
    private config: CryptoConfigEntry;
    private derivedKey: CryptoKey;

    constructor(private configName: string) {

        const configs = Client.u(__STORE_Crypto__);

        this.config = configs[configName];
    }

    /*
    **
    **
    */
    public async initKey() : Promise<void> {

        const password = new TextEncoder().encode(this.config.rawKey);
        const hashedPassword = await this.lib.digest('SHA-256', password);
        
        const baseKey = await this.lib.importKey('raw', hashedPassword, 'PBKDF2', false, ['deriveKey']);

        const derivationAlgorithm = {
            name: 'PBKDF2',
            salt: new Uint8Array(this.config.derivedKeySalt),
            iterations: this.config.derivedKeyIterations,
            hash: 'SHA-256',
        };

        const derivedKeyAlgorithm = {
            name: 'AES-CBC', 
            length: this.config.derivedKeyByteLength * 8
        };

        const derivedKey = await this.lib.deriveKey(derivationAlgorithm, baseKey, derivedKeyAlgorithm, true, ['encrypt', 'decrypt']);

        this.derivedKey = derivedKey;
    }

    /*
    **
    **
    */
    public async encrypt(input: ArrayBuffer) : Promise<ArrayBuffer> {

        const iv = Crypto.getRandomBytes(this.config.ivByteLength);

        const algorithm = {
            name: 'AES-CBC',
            iv: iv
        }

        const encrypted = await this.lib.encrypt(algorithm, this.derivedKey, input);

        const output = new Uint8Array(iv.byteLength + encrypted.byteLength)
        output.set(new Uint8Array(iv), 0);
        output.set(new Uint8Array(encrypted), iv.byteLength);

        return output;
    }

    /*
    **
    **
    */
    public async decrypt(input: ArrayBuffer) : Promise<ArrayBuffer> {

        const algorithm = {
            name: 'AES-CBC',
            iv: input.slice(0, this.config.ivByteLength)
        }

        return await this.lib.decrypt(algorithm, this.derivedKey, input.slice(this.config.ivByteLength));
    }

    /*
    **
    **
    */
    static getRandomBytes(length: number) : Uint8Array {

        const lib_ = self.crypto;
        const QUOTA = 65536

        const output = new Uint8Array(length);

        for (let i = 0; i < length; i += QUOTA) 
            lib_.getRandomValues(output.subarray(i, i + Math.min(length - i, QUOTA)));
        
        return output;
    }

    /*
    **
    **
    */
    static arrayBufferToBase64(input: ArrayBuffer) : string {

        let base64 = '';

        const encodings = Crypto.BASE64_ENCODINGS;
        const bytes = new Uint8Array(input);
        const byteLength = bytes.byteLength;
        const byteRemainder = byteLength % 3;
        const mainLength = byteLength - byteRemainder;
        
        let chunk;
        let a, b, c, d;
      
        for (let i = 0; i < mainLength; i = i + 3) {

            chunk = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
            a = (chunk & 16515072) >> 18;
            b = (chunk & 258048)   >> 12;
            c = (chunk & 4032)     >>  6;
            d = chunk & 63;
            base64 += encodings[a] + encodings[b] + encodings[c] + encodings[d];
        }
      
        if (byteRemainder == 1) {

            chunk = bytes[mainLength];
            a = (chunk & 252) >> 2;
            b = (chunk & 3)   << 4;
            base64 += encodings[a] + encodings[b] + '==';

        } else if (byteRemainder == 2) {

            chunk = (bytes[mainLength] << 8) | bytes[mainLength + 1];
            a = (chunk & 64512) >> 10;
            b = (chunk & 1008)  >>  4;
            c = (chunk & 15)    <<  2;
            base64 += encodings[a] + encodings[b] + encodings[c] + '=';
        }
        
        return base64;
    }

    /*
    **
    **
    */
    static base64ToArrayBuffer(input: string) : ArrayBuffer {
        
        const encodings = Crypto.BASE64_ENCODINGS;

        const removePaddingChars = function(input_) {

            const lkey = encodings.indexOf(input_.charAt(input_.length - 1));

            if (lkey == 64)
                return input_.substring(0,input_.length - 1);
    
            return input_;
        }

		input = removePaddingChars(input);
		input = removePaddingChars(input);

		const bytes = (input.length / 4) * 3;
		
		let uarray = new Uint8Array(bytes);
		let chr1, chr2, chr3;
		let enc1, enc2, enc3, enc4;
		let i = 0;
		let j = 0;
		
		input = input.replace(/[^A-Za-z0-9\+\/\=]/g, "");
		
		for (i=0; i<bytes; i+=3) {

			enc1 = encodings.indexOf(input.charAt(j++));
			enc2 = encodings.indexOf(input.charAt(j++));
			enc3 = encodings.indexOf(input.charAt(j++));
			enc4 = encodings.indexOf(input.charAt(j++));
	
			chr1 = (enc1 << 2) | (enc2 >> 4);
			chr2 = ((enc2 & 15) << 4) | (enc3 >> 2);
			chr3 = ((enc3 & 3) << 6) | enc4;
	
			uarray[i] = chr1;			
			if (enc3 != 64) uarray[i+1] = chr2;
			if (enc4 != 64) uarray[i+2] = chr3;
		}

		return uarray;	
	}

    /*
    **
    **
    */
    static async computePBKDF2HexHash(password: string, saltHex: string, iterations: number, hashByteLength, hashAlgorithm) : Promise<string> {

        const saltBuffer = Crypto.hexStringToArrayBuffer(saltHex);
    
        const passwordBuffer = new TextEncoder().encode(password);
    
        const importedKey = await window.crypto.subtle.importKey('raw', passwordBuffer, 'PBKDF2', false, ['deriveBits']);
    
        const hashBuffer = await window.crypto.subtle.deriveBits({
            name: 'PBKDF2',
            salt: saltBuffer,
            iterations: iterations,
            hash: hashAlgorithm,
        }, importedKey, hashByteLength*8);
    
        return Crypto.arrayBufferToHexString(hashBuffer);
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