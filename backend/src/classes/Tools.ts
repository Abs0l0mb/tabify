'use strict';

import * as nodePath from 'path';
import * as fs from 'fs';
import * as crypto from 'crypto';
import * as nanoid from 'nanoid';
import * as zlib from 'zlib';

import { execSync } from 'child_process';

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
    static sha256(data: string | Buffer) : string {

        return crypto.createHash('sha256').update(data).digest('hex');
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

        return new Promise(resolve => {
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

        return nanoid.customAlphabet(alphabet, size)();
    }

    /*
    **
    **
    */
    static randStr(size = 32) : string {

        return Tools.uid(size);
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
    static hex2bin(str: string) : string {

        return Buffer.from(str, 'hex').toString('utf8');
    }

    /*
    **
    **
    */
    static bin2hex(str: string) : string {

        return Buffer.from(str, 'utf8').toString('hex');
    }

    /*
    **
    **
    */
    static base64Encode(string: any) : string {
        
        return Buffer.from(string).toString('base64');
    }

    /*
    **
    **
    */
    static base64Decode(string: any) : string {
        
        return Buffer.from(string, 'base64').toString();
    }

    /*
    **
    **
    */
    static brotliCompress(input: Buffer) : Promise<Buffer> {

        return new Promise(resolve => {

            const compressor = zlib.createBrotliCompress();

            let buffer = Buffer.alloc(0);
            
            compressor.on('data', chunk => {
                buffer = Buffer.concat([buffer, chunk]);
            });

            compressor.on('end', () => {
                resolve(buffer);
            });

            compressor.write(input);
            compressor.end();
        });
    }

    /*
    **
    **
    */
    static bufferToArrayBuffer(buffer: Buffer) : ArrayBuffer {

        const arrayBuffer = new ArrayBuffer(buffer.length);
        const view = new Uint8Array(arrayBuffer);

        for (let i = 0; i < buffer.length; ++i)
          view[i] = buffer[i];

        return arrayBuffer;
    }

    /*
    **
    **
    */
    static arrayBufferToBuffer(arrayBuffer: ArrayBuffer) : Buffer {
        
        const buffer = Buffer.alloc(arrayBuffer.byteLength);
        const view = new Uint8Array(arrayBuffer);

        for (let i = 0; i < buffer.length; ++i) 
            buffer[i] = view[i];
        
        return buffer;
    }

    /*
    **
    **
    */
    static getMime(buffer: Buffer) : string | null {

        //require magika (python version):
        //  $ sudo apt update
        //  $ sudo apt install python3 python3-pip
        //  $ sudo -H pip install magika

        let mime: string;

        const filePath = nodePath.resolve('/tmp', Tools.uid());
        fs.writeFileSync(filePath, buffer);

        const output = execSync(`magika --json "${filePath}"`).toString();

        fs.rmSync(filePath);

        try {
            
            const result = JSON.parse(output);

            if (Array.isArray(result) 
             && result.length === 1 
             && typeof result[0] === 'object' 
             && result[0] !== null
             && typeof result[0].output === 'object'
             && result[0].output !== null
             && typeof result[0].output.score === 'number'
             && result[0].output.score > 0.9
             && typeof result[0].output.mime_type === 'string')
                return result[0].output.mime_type;
            else
                return null;

        } catch(error) {
            return null;
        }
    }

    /*
    **
    **
    */
    static isSafe(buffer: Buffer) : boolean {

        //require clamav:
        //  $ sudo apt update
        //  $ sudo apt install clamav

        const filePath = nodePath.resolve('/tmp', Tools.uid());
        fs.writeFileSync(filePath, buffer);

        const output = Tools.parseClamscanOutput(execSync(`clamscan "${filePath}"`));

        fs.rmSync(filePath);
        
        return output.infected_files === 0;
    }   

    /*
    **
    **
    */
    static parseClamscanOutput(raw: Buffer) : any {

        const output = {};

        const summary = raw.toString().split('SCAN SUMMARY')[1].split('\n');
        summary.shift();
        summary.pop();

        for (const entry of summary) {

            const split = entry.split(':');

            const key = split[0].toLowerCase().replace(/ /g, '_');

            let value: number | string = split[1].trim();

            const number = new Number(value).valueOf();
            if (!isNaN(number))
                value = number;

            output[key] = value;
        }

        return output;
    }
}