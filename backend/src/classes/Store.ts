'use strict';

import * as fs from 'fs';
import * as nodePath from 'path';

export class Store {

    static DATA: any = {};

    static DIRECTORY_PATH: string = nodePath.resolve(process.cwd(), 'store'); 

    static cache: any = {};

    /*
    **
    **
    */
    static getFilePath(key: string) : string {
    
        return `${Store.DIRECTORY_PATH}/${key}.json`;
    }

    /*
    **
    **
    */
    static get(key: string) : any {

        if (typeof Store.cache[key] === 'object')
            return Store.cache[key];
        else {

            let filePath = Store.getFilePath(key);

            if (!fs.existsSync(filePath))
                return null;
                
            let data = JSON.parse(fs.readFileSync(filePath).toString());
            
            Store.cache[key] = data;

            return data;
        }
    }

    /*
    **
    **
    */
    static async set(key: string, value: any) : Promise<void> {

        let filePath = Store.getFilePath(key);

        let data = JSON.stringify(value);

        fs.writeFileSync(filePath, data);

        Store.cache[key] = data;
    }
}