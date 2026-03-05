'use strict';

export class Log {

    public static readonly Reset = "\x1b[0m";
    public static readonly Bright = "\x1b[1m";
    public static readonly Dim = "\x1b[2m";
    public static readonly Underline = "\x1b[4m";
    public static readonly Blink = "\x1b[5m";
    public static readonly Reverse = "\x1b[7m";
    public static readonly Hidden = "\x1b[8m";

    public static readonly FgBlack = "\x1b[30m";
    public static readonly FgRed = "\x1b[31m";
    public static readonly FgGreen = "\x1b[32m";
    public static readonly FgYellow = "\x1b[33m";
    public static readonly FgBlue = "\x1b[34m";
    public static readonly FgMagenta = "\x1b[35m";
    public static readonly FgCyan = "\x1b[36m";
    public static readonly FgWhite = "\x1b[37m";

    public static readonly BgBlack = "\x1b[40m";
    public static readonly BgRed = "\x1b[41m";
    public static readonly BgGreen = "\x1b[42m";
    public static readonly BgYellow = "\x1b[43m";
    public static readonly BgBlue = "\x1b[44m";
    public static readonly BgMagenta = "\x1b[45m";
    public static readonly BgCyan = "\x1b[46m";
    public static readonly BgWhite = "\x1b[47m";

    /*
    **
    **
    */
    public static log(...args: any[]) : void {

        console.log(`${Log.Dim}[${new Date().toLocaleString('fr', { timeZone: process.env.TIME_ZONE })}] @${process.pid}${Log.Reset} [${process.env.LOG_ID}]${Log.Reset}`, ...args);
    }

    /*
    **
    **
    */
    public static blank() : void {

        Log.log('');
    }

    /*
    **
    **
    */
    public static green(message: string) : void {

        Log.log(`${Log.FgGreen}${Log.Bright}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    public static red(message: string) : void {

        Log.log(`${Log.FgRed}${Log.Bright}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    public static magenta(message: string) : void {

        Log.log(`${Log.FgMagenta}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    public static yellow(message: string) : void {

        Log.log(`${Log.FgYellow}${Log.Bright}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    public static printError(error: any) : void {

        if (typeof error !== 'object' || error === null || !error.stack)
            return;

        error.stack.split('\n').forEach((line: string) => {
            Log.log(`${Log.FgRed}${line}${Log.Reset}`);
        });
    }
}