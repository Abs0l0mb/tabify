'use strict';

import { Store } from '@src/classes';

import * as crypto from 'crypto';

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

export class Crypto {

    private config: CryptoConfigEntry;
    private derivedKey: crypto.CipherKey;

    constructor(private configName: string) {

        const configs: CryptoConfigs = Store.get('crypto');

        if (configs && configs[configName]) {
            this.config = configs[configName];
            this.initKey();
        }
        else 
            throw new Error(`Config "${configName}" not found in crypto configs`)
    } 

    /*
    **
    **
    */
    private initKey() : void {

        const password = Buffer.from(this.config.rawKey);
        const hashedPassword = crypto.createHash('sha256').update(password).digest();

        this.derivedKey = crypto.pbkdf2Sync(hashedPassword, Buffer.from(this.config.derivedKeySalt), this.config.derivedKeyIterations, this.config.derivedKeyByteLength, 'sha256');
    }

    /*
    **
    **
    */
    public encrypt(value: Buffer) : Buffer {

        const iv = crypto.randomBytes(this.config.ivByteLength);
        const cipher = crypto.createCipheriv('aes-256-cbc', this.derivedKey, iv);
        const encrypted = Buffer.concat([cipher.update(value), cipher.final()]);
        
        return Buffer.concat([iv, encrypted]);
    }

    /*
    **
    **
    */
    public decrypt(buffer: Buffer) : Buffer {

        const iv = buffer.subarray(0, this.config.ivByteLength);
        const encrypted = buffer.subarray(this.config.ivByteLength);
        const decipher = crypto.createDecipheriv('aes-256-cbc', this.derivedKey, iv);
        const decrypted = Buffer.concat([decipher.update(encrypted), decipher.final()]);

        return decrypted;
    }
}