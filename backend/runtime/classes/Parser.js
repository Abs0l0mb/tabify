'use strict';

export class Parser {

    static parse(data, map, label) {
        
        if (typeof label === 'string' && label.trim().length > 1)
            label = label + ' ';
        else
            label = '';
            
        let output = {};

        for (let key in map) {

            let parser = map[key];
            let optional = false;

            if (Array.isArray(parser) && parser.length === 2 && parser[1] === 'optional') {
                parser = parser[0];
                optional = true;
            }
            
            if (typeof parser !== 'function')
                continue;

            if (typeof data[key] === 'undefined') {

                if (optional)
                    continue;
                else
                    throw new Error(`${label}${key}@not-found`);
            }

            if (optional && data[key] === "")
                continue;

            if (data[key] === null) {
                output[key] = null;
                continue;
            }

            let result = parser(data[key], true);

            if (Array.isArray(result)) {

                if (result.length === 2 && result[0] === true)
                    output[key] = result[1];
                else if (result.length === 2 && result[0] === false && typeof result[1] === 'string')
                    throw new Error(`${label}${key}@${result[1]}`);
                else
                    throw new Error(`${label}${key}@parsing-result-error`);
            }
            else {

                if (result === true)
                    output[key] = data[key];
                else if (result)
                    output[key] = result;
                else
                    throw new Error(`${label}${key}@not-valid`);
            }
        }

        return output;
    }

    /*
    **
    **
    */
    static string(input) {

        if ((typeof input === 'string' || typeof input === 'number'))
            return [true, input.toString()];
        
        return [false, 'string-error'];
    }

    /*
    **
    **
    */
    static number(input) {

        if (typeof input === 'number')
            return [true, input];
    
        return [false, 'number-error'];
    }

    /*
    **
    **
    */
    static float(input) {

        if (/^[+-]?\d+(\.\d+)?$/.test(input))
            return [true, parseFloat(input)];
        
        return [false, 'float-error'];
    }
    
    /*
    **
    **
    */
    static integer(input) {

        if (/^\d+$/.test(input))
            return [true, parseInt(input)];
    
        return [false, 'integer-error'];
    }

    /*
    **
    **
    */
    static boolean(input) {

        if (input === true || input === false)
            return [true, input];
        else
            return [false, 'boolean-error'];
    }

    /*
    **
    **
    */
    static object(input) {

        if (typeof input === 'object' && input !== null)
            return [true, input];
        else
            return [false, 'object-error'];
    }

    /*
    **
    **
    */
    static array(input) {

        if (Array.isArray(input))
            return [true, input];
        else
            return [false, 'array-error'];
    }
}