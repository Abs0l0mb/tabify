'use strict';

import * as nodePath from 'path';
import * as fs from 'fs';

export class Tools {
    
    /*
    **
    **
    */
    static copyDirSync(source, target) {

        let files = [];
    
        const targetFolder = nodePath.join(target, nodePath.basename(source));

        if (!fs.existsSync(targetFolder))
            fs.mkdirSync(targetFolder);
    
        if (fs.lstatSync(source).isDirectory()) {
            files = fs.readdirSync(source);
            files.forEach(function(file) {
                let curSource = nodePath.join(source, file);
                if (fs.lstatSync(curSource).isDirectory()) 
                    this.copyDirSync(curSource, targetFolder);
                else
                    fs.copyFileSync(curSource, `${targetFolder}/${nodePath.basename(curSource)}`);
            }.bind(this));
        }
    }

    /*
    **
    **
    */
    static rmDirSync(nodePath) {

        if (fs.existsSync(nodePath)) {

            fs.readdirSync(nodePath).forEach(function(file,index) {
                const curPath = nodePath + "/" + file;
                if (fs.lstatSync(curPath).isDirectory())
                    this.rmDirSync(curPath);
                else
                    fs.unlinkSync(curPath);
            }.bind(this));

            fs.rmdirSync(nodePath);
        }
    }
    
    /*
    **
    **
    */
    static randStr(length=8) {

        let text = "";
        const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (let i = 0; i < length; i++)
            text += possible.charAt(Math.floor(Math.random() * possible.length));

        return text;
    }
    
    /*
    **
    **
    */
    static sleep(time) {

        return new Promise(function(resolve) {
            setTimeout(resolve, time);
        });
    }

    /*
    **
    **
    */
    static getArgumentValue(key) {

        for (let arg of process.argv) {
            if (arg.slice(0, key.length + 1) === `${key}=`)
                return arg.slice(key.length + 1);
        }

        return undefined;
    }

    /*
    **
    **
    */
    static humanSince(milliseconds) {

        const units = [
            { label: 'y', ms: 1000 * 60 * 60 * 24 * 365 },
            { label: 'mo', ms: 1000 * 60 * 60 * 24 * 30 },
            { label: 'd', ms: 1000 * 60 * 60 * 24 },
            { label: 'h', ms: 1000 * 60 * 60 },
            { label: 'm', ms: 1000 * 60 },
            { label: 's', ms: 1000 },
            { label: 'ms', ms: 1 }
        ];
    
        const result = [];
    
        for (let i = 0; i < units.length; i++) {

            if (milliseconds >= units[i].ms) {

                const timeValue = Math.floor(milliseconds / units[i].ms);
                result.push(`${timeValue}${units[i].label}`);
                
                milliseconds -= timeValue * units[i].ms;
            }
    
            if (result.length === 2) break;
        }
    
        if (result.length === 0 && milliseconds > 0)
            result.push(`${milliseconds}ms`);
    
        return result.join(' ');
    }
}