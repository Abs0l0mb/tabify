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

        return Parser.parse(JSON.parse(fs.readFileSync(nodePath.resolve(process.cwd(), 'runtime.json'))), {
            runtimeName: Parser.string,
            appName: Parser.string,
            themeColor: Parser.string,
            robotsOnProd: Parser.boolean,
            domain: Parser.string,
            'og:type': Parser.string,
            'twitter:site': Parser.string,
            extraHead: Parser.string,
            devPort: Parser.integer,
            vendorDependencies: Parser.array,
            pages: Parser.array
        });
    }
}