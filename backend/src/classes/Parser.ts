'use strict';

import { PublicError } from '@src/classes';

export class Parser {

    /*
    **
    **
    */
    static async parse(data: any, map: any, publicErrors: boolean = true) : Promise<any> {
        
        let errorConstructor = publicErrors ? PublicError : Error;

        if (!data || !map)
            throw new errorConstructor('parser-data-error');

        let output: any = {};

        for (let key in map) {

            let parser = map[key];
            let optional = false;

            if (Array.isArray(parser) && parser.length === 2 && parser[1] === 'optional') {
                parser = parser[0];
                optional = true;
            }
            
            if (typeof parser !== 'function')
                continue;

            if (typeof data[key] === 'undefined' || data[key] === null || (typeof data[key] === 'string' && data[key].trim() === '')) {

                if (optional && (typeof data[key] === 'string' && data[key].trim() === '')) {
                    output[key] = '';
                    continue;
                }
                else if (optional && data[key] === null) {
                    output[key] = null;
                    continue;
                }
                else if (optional)
                    continue;
                else
                    throw new errorConstructor(`${key}@not-found`);
            }

            let result = await parser(data[key], true);

            if (Array.isArray(result)) {

                if (result.length === 2 && result[0] === true)
                    output[key] = result[1];
                else if (result.length === 2 && result[0] === false && typeof result[1] === 'string')
                    throw new errorConstructor(`${key}@${result[1]}`);
                else
                    throw new errorConstructor(`${key}@unexpected-not-valid`);
            }
            else {

                if (result === true)
                    output[key] = data[key];
                else if (result)
                    output[key] = result;
                else
                    throw new errorConstructor(`${key}@not-valid`);
            }
        }

        return output;
    }

    /*
    **
    **
    */
    static string(input: any) : [boolean, any] {

        if (typeof input === 'string' || typeof input === 'number')
            return [true, input.toString()];
        
        return [false, 'string-error'];
    }

    /*
    **
    **
    */
    static address(input: any) : [boolean, any] {

        if ((typeof input === 'string' || typeof input === 'number') 
            && input.toString().length >= 5 
            && input.toString().length <= 100) {

            return [true, input.toString()];
        }
        
        return [false, 'address-error'];
    }

    /*
    **
    **
    */
    static number(input: any) : [boolean, any] {

        if (typeof input === 'number')
            return [true, input];
    
        return [false, 'number-error'];
    }

    /*
    **
    **
    */
    static float(input: any) : [boolean, any] {

        if (/^[+-]?\d+(\.\d+)?$/.test(input))
            return [true, parseFloat(input)];
        
        return [false, 'float-error'];
    }
    
    /*
    **
    **
    */
    static integer(input: any) : [boolean, any] {

        if (/^\d+$/.test(input))
            return [true, parseInt(input)];
    
        return [false, 'integer-error'];
    }

    /*
    **
    **
    */
    static boolean(input: any) : [boolean, any] {

        if (input === true || input === false)
            return [true, input];
        else
            return [false, 'boolean-error'];
    }

    /*
    **
    **
    */
    static object(input: any) : [boolean, any] {

        if (typeof input === 'object' && input !== null)
            return [true, input];
        else {
        
            try {
    
                let data = JSON.parse(input);
    
                if (typeof data === 'object' && data !== null)
                    return [true, data];
    
            } catch(error) {
                
                return [false, 'object-error'];
            }
        }

        return [false, 'object-error'];
    }

    /*
    **
    **
    */
    static percent(input: any) : [boolean, any] {

        if (/^[+-]?\d+(\.\d+)?$/.test(input) && parseFloat(input) >= 0)
            return [true, parseFloat(input)];
    
        return [false, 'percent-error'];
    }

    /*
    **
    **
    */
    static integerArray(input: any) : [boolean, any] {

        try {

            const output: any[] = [];
            
            for (let value of input) {

                if (!/^\d+$/.test(value))
                    return [false, 'integer-array-error'];
                
                output.push(parseInt(value));
            }

            return [true, output];

        } catch(error) {
            
            return [false, 'integer-array-error'];
        }
    }

    /*
    **
    **
    */
    static hex(input: any) : [boolean, any] {

        if (/^[A-Fa-f0-9]+$/.test(input))
            return [true, input];
        else
            return [false, 'hex-error'];
    }

    /*
    **
    **
    */
    static hexColor(input: any) : [boolean, any] {

        if (/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(input))
            return [true, input]
        else
            return [false, 'hex-color-error'];
    }

    /*
    **
    **
    */
    static sha256(input: any) : [boolean, any] {

        if (input && /^[0-9a-fA-F]{64}$/.test(input))
            return [true, input];
        else
            return [false, 'sha256-error'];
    }

    /*
    **
    **
    */
    static phone(input: any) : [boolean, any] {

        if ((typeof input !== 'string' && typeof input !== 'number') || input.toString().length > 25)
            return [false, 'phone-error']
            
        if (/^([0-9 +\-()]+)$/.test(input.toString()))
            return [true, input.toString()];
    
        return [false, 'phone-error'];
    }

    /*
    **
    **
    */
    static email(input: any) : [boolean, any] {

        if (typeof input !== 'string' || input.length > 254)
            return [false, 'email-error'];

        if (/^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
            .test(input))
            return [true, input];
    
        return [false, 'email-error'];
    }

    /*
    **
    **
    */
    static password(input: any) : [boolean, any] {

        if (typeof input === 'string' && input.length > 6)
            return [true, input];
        else
            return [false, 'password-error'];
    }

    /*
    **
    **
    */
    static true(input: any) : [boolean, any] {
        
        if (input !== true)
            return [false, 'true-expected'];
        else
            return [true, input];
    }

    /*
    **
    **
    */
    static date(input: any) : [boolean, any] {

        if (isNaN(Date.parse(input)))
            return [false, 'date-error'];
        else
            return [true, new Date(input)];
    }

    /*
    **
    **
    */
    static classKey(input: any) : [boolean, any] {

        if ((typeof input === 'string' || typeof input === 'number') && input.toString().match(/^[a-z0-9-]+$/g))
            return [true, input.toString()];
        
        return [false, 'format-error'];
    }
}