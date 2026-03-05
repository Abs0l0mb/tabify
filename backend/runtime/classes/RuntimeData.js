'use strict';

import { Parser } from './Parser.js';
import * as fs from 'fs';
import * as nodePath from 'path';

export class RuntimeData {

    /*
    **
    **
    */
    static getPackage() {

        return JSON.parse(fs.readFileSync(nodePath.resolve(process.cwd(), 'package.json')));
    }

    /*
    **
    **
    */
    static getOptions() {

        const options = Parser.parse(JSON.parse(fs.readFileSync(nodePath.resolve(process.cwd(), 'runtime.json'))), {
            runtimeName: Parser.string,
            mainProcess: Parser.object,
            subProcesses: Parser.array
        }, 'runtime.json');
 
        options.mainProcess = Parser.parse(options.mainProcess, {
            distName: Parser.string,
            entryPoint: Parser.string,
            devPort: Parser.integer
        }, 'runtime.json > mainProcess');

        for (const key in options.subProcesses) {
        
            options.subProcesses[key] = Parser.parse(options.subProcesses[key], {
                distName: Parser.string,
                entryPoint: Parser.string,
                instanciationKey: Parser.string
            }, `runtime.json > subProcesses[${key}]`);
        }

        return options;
    }
}