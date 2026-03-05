'use strict';

import {
    Div,
    ClientLocation,
    Menu,
    Tools
} from '@src/classes';

export class Navigation extends Div {

    private menu: Menu;
    private reduced: boolean = false;

    constructor() {
        
        super('navigation');

        this.draw();
        
        ClientLocation.get().api.on('connected', this.onConnected.bind(this));
        ClientLocation.get().api.on('not-connected', this.onNotConnected.bind(this));

        setTimeout(function() {
            this.setData('displayed', 1);
        }.bind(this), 50);
    }

    /*
    **
    **
    */
    private async drawMenu() : Promise<void> {

        if (this.menu)
            this.menu.empty();

        this.menu = new Menu([

            { label: 'Team', extraClass: 'settings team', submenuOpened: true, submenu: [
                { label: 'Tasks', path: '/tasks', extraClass: 'tasks' },
                { label: 'Archived', path: '/archived', extraClass: 'archived' },
                { label: 'Categories', path: '/categories', extraClass: 'categories' },
                { label: 'Documents', path: '/documents', extraClass: 'document' },
                { label: 'Members', path: '/team', extraClass: 'members' },
                { label: 'Progress report', path: '/progress-report', extraClass: 'progress-report', requiredRight: 'PROGRESS REPORT' }
            ]},
            
            { label: 'Administration', extraClass: 'settings', submenu: [
                { label: 'Teams', path: '/settings/teams', extraClass: 'teams', requiredRight: 'MANAGE TEAMS' }
            ]},

            { label: 'Account', path: '/me', extraClass: 'my-account' }
        ], this);

        await this.menu.draw();

        this.menu.on('item-click', () => {
            this.setMobileMenuVisibility(false);
        });

        if (this.getNavigationReducedSetting())
            this.reduce();
    }

    /*
    **
    **
    */
    private async onConnected() : Promise<void> {

        await this.drawMenu();

        this.setMobileMenuVisibility(false);
    }
    
    /*
    **
    **
    */
    private async onNotConnected() : Promise<void> {

        this.setMobileMenuVisibility(false);

        if (this.menu) {
            await Tools.sleep(350);
            this.menu.delete();
        }
    }

    /*
    **
    **
    */
    private draw() : void {

        const background = new Div('background', this).onNative('click', () => {
            this.setMobileMenuVisibility(false);
        });

        new Div('mask', background).onNative('click', async () => {
            if (this.reduced)
                this.enlarge();
            else
                this.reduce();
        });

        new Div('mobile-button menu-button', this).onNative('click', () => {
            parseInt(this.getData('mobile-menu-displayed')) === 1 ? this.setMobileMenuVisibility(false) : this.setMobileMenuVisibility(true);
        });

        new Div('logo', this)
        .onNative('click', () => {
            ClientLocation.get().router.routeFirstPath()
        });
    }
    
    /*
    **
    **
    */
    private setMobileMenuVisibility(visibility: boolean) : void {

        this.setData('mobile-menu-displayed', visibility ? 1 : 0);
    }

    /*
    **
    **
    */
    private getNavigationReducedSetting() : boolean {

        const setting = ClientLocation.get().getSetting('navigation-reduced');

        if (typeof setting === 'boolean')
            return setting;
        else {
            this.setNavigationReducedSetting(false);
            return false;
        }
    }

    /*
    **
    **
    */
    private setNavigationReducedSetting(value: boolean) : void {

        ClientLocation.get().setSetting('navigation-reduced', value);
    }

    /*
    **
    **
    */
    private reduce() : void {

        ClientLocation.get().block.setData('navigation-reduced', 1);
        this.reduced = true;
        this.menu.reduce();

        this.setNavigationReducedSetting(true);
    }

    /*
    **
    **
    */
    private enlarge() : void {

        ClientLocation.get().block.setData('navigation-reduced', 0);
        this.reduced = false;
        this.menu.enlarge();

        this.setNavigationReducedSetting(false);
    }

    /*
    **
    **
    */
    public async refresh() : Promise<void> {

        this.menu.refresh();
    }
}